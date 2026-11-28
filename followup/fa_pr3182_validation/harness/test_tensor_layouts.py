# Tensor-layout correctness for flash_attn.ops.triton.cross_entropy.cross_entropy_loss (public API).
# Compares loss and dlogits with a PyTorch fp32 reference. Stock is expected to fail the strided cases.
import os, itertools, pytest, torch, torch.nn.functional as F
import fa_noext  # noqa: F401  (registers flash_attn from FA_SRC)
import flash_attn.ops.triton.cross_entropy as ce

DEV = "cuda"
FA_SRC = os.environ["FA_SRC"]
assert ce.__file__.startswith(FA_SRC), (ce.__file__, FA_SRC)

class Count:
    """Proxy for a Triton kernel: counts launches, forwards everything else."""
    def __init__(self, k): self.k, self.n = k, 0
    def __getitem__(self, grid):
        self.n += 1
        return self.k[grid]
    def __getattr__(self, a): return getattr(self.k, a)

@pytest.fixture(autouse=True)
def count_kernels(monkeypatch):
    f, b = Count(ce.cross_entropy_fwd_kernel), Count(ce.cross_entropy_bwd_kernel)
    monkeypatch.setattr(ce, "cross_entropy_fwd_kernel", f)
    monkeypatch.setattr(ce, "cross_entropy_bwd_kernel", b)
    yield (f, b)

def make_labels(n, V, stride, offset, dtype, mixed_ignore, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    logical = torch.randint(0, V, (n,), device=DEV, generator=g)
    if mixed_ignore:
        logical = torch.where(torch.rand(n, device=DEV, generator=g) < 0.3, torch.full_like(logical, -100), logical)
    if stride == 0:  # expanded
        base = logical[:1].clone()
        return base.to(dtype).expand(n), base.expand(n).clone()
    size = offset + (n - 1) * stride + 1 + 3
    store = torch.randint(0, V, (size,), device=DEV, generator=g).to(dtype)  # filler = valid wrong ids
    view = store[offset:][::stride][:n] if stride > 1 else store[offset:offset + n]
    view.copy_(logical.to(dtype))
    assert view.shape == (n,) and view.stride(0) == max(stride, 1) and view.storage_offset() == offset
    return view, logical

def reference(x, y_logical, smoothing, scale, lse_sq, ignore=-100):
    xf = x.detach().float().clone().requires_grad_()
    s = xf * scale
    loss = F.cross_entropy(s, y_logical.long(), reduction="none", label_smoothing=smoothing, ignore_index=ignore)
    z = lse_sq * torch.logsumexp(s, -1).square()
    loss = loss + torch.where(y_logical != ignore, z, torch.zeros_like(z))
    return loss, xf

def run_case(n, V, xdt, ldt, stride, offset, smoothing=0.0, scale=1.0, lse_sq=0.0, mixed=False, seed=0, count=None):
    torch.manual_seed(seed)
    x = torch.randn(n, V, device=DEV, dtype=xdt, requires_grad=True)
    y, y_logical = make_labels(n, V, stride, offset, ldt, mixed, seed + 1)
    loss_ref, xref = reference(x, y_logical, smoothing, scale, lse_sq)
    loss, _ = ce.cross_entropy_loss(x, y, label_smoothing=smoothing, logit_scale=scale, lse_square_scale=lse_sq)
    g = torch.randn(n, device=DEV, generator=torch.Generator(device=DEV).manual_seed(seed + 2))
    loss_ref.backward(g); loss.backward(g)
    rtol, atol = (1e-5, 1e-6) if xdt == torch.float32 else (1e-3, 1e-4)
    # forward and backward are checked separately so a failure says which one broke
    fwd_ok = torch.allclose(loss, loss_ref, rtol=rtol, atol=atol)
    bwd_ok = torch.allclose(x.grad.float(), xref.grad.to(xdt).float() if xdt != torch.float32 else xref.grad, rtol=rtol, atol=atol)
    if count is not None:
        f, b = count
        assert f.n == 1 and b.n == 1, f"Triton kernels launched fwd={f.n} bwd={b.n} (expected 1 each)"
    return fwd_ok, bwd_ok, (loss - loss_ref).abs().max().item(), (x.grad.float() - xref.grad.to(xdt).float()).abs().max().item() if xdt != torch.float32 else (x.grad - xref.grad).abs().max().item()

DT = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
LAYOUTS = [("contig", 1, 0), ("off1", 1, 1), ("off2", 1, 2), ("off3", 1, 3),
           ("s2", 2, 0), ("s3", 3, 0), ("s4", 4, 0), ("s7", 7, 0), ("s16", 16, 0),
           ("s2o1", 2, 1), ("s3o2", 3, 2), ("s4o3", 4, 3), ("s7o1", 7, 1), ("s16o5", 16, 5), ("expand", 0, 0)]
CASES = []
# A: every layout x every logits dtype at a mid size
for (ln, st, off), dn in itertools.product(LAYOUTS, DT):
    CASES.append((f"A-{ln}-{dn}", dict(n=129, V=1025, xdt=DT[dn], ldt=torch.long, stride=st, offset=off)))
# B: batch sizes x layouts (contig, s2, s3o2, expand)
for n, (ln, st, off) in itertools.product([1, 2, 3, 63, 64, 65, 127, 128, 129, 255, 256, 257, 1000], [LAYOUTS[0], LAYOUTS[4], LAYOUTS[10], LAYOUTS[14]]):
    CASES.append((f"B-n{n}-{ln}", dict(n=n, V=1025, xdt=torch.float16, ldt=torch.long, stride=st, offset=off)))
# C: vocab sizes around block boundaries x layouts (BLOCK_SIZE = next pow2 up to 16K fwd, 4K bwd)
for V, (ln, st, off) in itertools.product([1, 2, 31, 32, 33, 1023, 1024, 1025, 4095, 4096, 4097, 16383, 16384, 16385, 50257, 128256], [LAYOUTS[0], LAYOUTS[4], LAYOUTS[10], LAYOUTS[14]]):
    CASES.append((f"C-V{V}-{ln}", dict(n=33 if V < 50000 else 17, V=V, xdt=torch.float16, ldt=torch.long, stride=st, offset=off)))
# D: loss options x layouts x ignore-index
for (ln, st, off), opts, mixed in itertools.product([LAYOUTS[0], LAYOUTS[4], LAYOUTS[10]],
        [dict(smoothing=0.1), dict(scale=0.7), dict(lse_sq=1e-2), dict(smoothing=0.1, scale=0.7, lse_sq=1e-2)], [False, True]):
    on = "-".join(f"{k}{v}" for k, v in opts.items())
    CASES.append((f"D-{ln}-{on}-ign{int(mixed)}", dict(n=257, V=4097, xdt=torch.bfloat16, ldt=torch.long, stride=st, offset=off, mixed=mixed, **opts)))
# E: ignore-index with all-ignored and none-ignored; int32 labels
for (ln, st, off) in [LAYOUTS[0], LAYOUTS[4], LAYOUTS[10]]:
    CASES.append((f"E-int32-{ln}", dict(n=129, V=1025, xdt=torch.float16, ldt=torch.int32, stride=st, offset=off, mixed=True)))
    CASES.append((f"E-ign-{ln}", dict(n=129, V=1025, xdt=torch.float16, ldt=torch.long, stride=st, offset=off, mixed=True, seed=5)))

@pytest.mark.parametrize("name,kw", CASES, ids=[c[0] for c in CASES])
def test_layout(name, kw, count_kernels):
    fwd_ok, bwd_ok, fe, be = run_case(count=count_kernels, **kw)
    assert fwd_ok and bwd_ok, f"fwd_ok={fwd_ok} (max err {fe:.3g}) bwd_ok={bwd_ok} (max err {be:.3g})"
