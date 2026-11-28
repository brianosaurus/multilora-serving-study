# One benchmark block for one tree in one process. Usage: bench_proc.py TREE MODE OUT.json   (MODE: contig | strided)
import os, sys, json, time, subprocess, statistics, itertools, torch
tree, mode, out = sys.argv[1:4]
os.environ["FA_SRC"] = f"/root/fa/{tree}" if tree != "stock2" else "/root/fa/stock2"
sys.path.insert(0, "/root/fa/harness")
import fa_noext  # noqa
import flash_attn.ops.triton.cross_entropy as ce
assert ce.__file__.startswith(os.environ["FA_SRC"]), ce.__file__
dev = "cuda"
def smi():
    r = subprocess.run(["nvidia-smi","--query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,clocks_throttle_reasons.active","--format=csv,noheader,nounits"],capture_output=True,text=True).stdout.strip()
    return {"smi": r, "loadavg": os.getloadavg()}
env0 = smi()
# warm the clocks (idle P8 is 180 MHz): sustained matmul for ~3 s
a = torch.randn(8192, 8192, device=dev, dtype=torch.float16); t0 = time.time()
while time.time() - t0 < 3: a @ a
torch.cuda.synchronize(); del a
def timeit(fn, warm=30, iters=100):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    s = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]; e = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        s[i].record(); fn(); e[i].record()
    torch.cuda.synchronize()
    return [a.elapsed_time(b) for a, b in zip(s, e)]
def summ(v):
    v = sorted(v); n = len(v)
    return {"median": statistics.median(v), "p10": v[n//10], "p90": v[(9*n)//10], "min": v[0], "max": v[-1], "raw": v}
if mode == "contig":
    layouts = [("contig", 1)]; cfgs = list(itertools.product([torch.float16, torch.bfloat16], [32000, 128256], [16, 512, 8192, 32768]))
else:
    layouts = [("s2", 2), ("s16", 16)]; cfgs = list(itertools.product([torch.float16], [32000, 128256], [512, 8192, 32768]))
res = []
for (ln, st), (dt, V, n) in itertools.product(layouts, cfgs):
    torch.manual_seed(0)
    x = torch.randn(n, V, device=dev, dtype=dt, requires_grad=True)
    store = torch.randint(0, V, (n * st,), device=dev)
    y = store[::st] if st > 1 else store
    assert y.stride(0) == st
    g = torch.randn(n, device=dev)
    def fwd():
        with torch.no_grad(): ce.cross_entropy_loss(x, y)
    loss, _ = ce.cross_entropy_loss(x, y)
    def bwd(): loss.backward(g, retain_graph=True)
    def both():
        l, _ = ce.cross_entropy_loss(x, y); l.backward(g)
    row = {"tree": tree, "layout": ln, "stride": st, "dtype": str(dt).split(".")[-1], "V": V, "n": n}
    torch.cuda.synchronize(); torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats(); both(); torch.cuda.synchronize()
    row["extra_peak_bytes_fwdbwd"] = torch.cuda.max_memory_allocated() - base
    x.grad = None
    row["fwd"] = summ(timeit(fwd)); row["bwd"] = summ(timeit(bwd)); row["fwdbwd"] = summ(timeit(both))
    x.grad = None
    res.append(row); del x, store, y, g, loss
    torch.cuda.empty_cache()
json.dump({"tree": tree, "mode": mode, "env_start": env0, "env_end": smi(), "rows": res}, open(out, "w"))
