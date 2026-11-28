# Balanced randomized blocks, one process (one GPU workload) at a time.
import random, subprocess, os, time, sys
os.makedirs("/root/fa/raw/bench", exist_ok=True)
log = open("/root/fa/raw/bench/driver.log", "a")
def say(s): log.write(time.strftime("%H:%M:%S ") + s + "\n"); log.flush()
def gpu_free():
    r = subprocess.run(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],capture_output=True,text=True).stdout.strip()
    return int(r) < 300
def block(trees, mode, nblocks, seed):
    rnd = random.Random(seed)
    for b in range(nblocks):
        order = trees[:]; rnd.shuffle(order)
        for t in order:
            assert gpu_free(), "GPU not free"
            out = f"/root/fa/raw/bench/{mode}_b{b}_{t}.json"
            say(f"start {mode} block {b} {t}")
            r = subprocess.run(["/root/fa/venv/bin/python", "/root/fa/harness/bench_proc.py", t, mode, out],
                               env={**os.environ, "TRITON_CACHE_DIR": f"/root/fa/tcache_bench_{t}_{mode}_{b}"}, capture_output=True, text=True, cwd=f"/root/fa/{t}")
            say(f"done {mode} block {b} {t} rc={r.returncode} {r.stderr[-300:] if r.returncode else ""}")
block(["stock", "stock2", "patched", "alt"], "contig", 10, 1)
block(["patched", "alt"], "strided", 6, 2)
say("BENCH_DONE")
