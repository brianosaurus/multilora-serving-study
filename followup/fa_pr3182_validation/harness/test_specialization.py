# Triton specialization: alternate label layouts within ONE process (shared kernel cache) in several orders.
import os, random, pytest, torch
import fa_noext  # noqa: F401
import flash_attn.ops.triton.cross_entropy as ce
from test_tensor_layouts import run_case, DT
FA_SRC = os.environ["FA_SRC"]; assert ce.__file__.startswith(FA_SRC)
C, S2, S3, S16, S4 = (1,0), (2,0), (3,2), (16,0), (4,0)   # (stride, storage_offset); S3 has an aligned offset (2 elements)
ORDERS = {
 "contig,strided,contig": [C, S2, C], "strided,contig,strided": [S2, C, S2],
 "multi-stride": [S2, S3, S16, S4, C, S3, S2, C, S16],
 "same-stride-diff-offset": [(2,0),(2,2),(2,4),(2,0),(2,2)],   # aligned offsets only (unaligned go through pad)
 "same-stride-diff-offset-incl-unaligned": [(2,0),(2,1),(2,2),(2,3),(2,0)],
 "expand-between": [C, (0,0), S2, (0,0), C],
}
rnd = random.Random(0); ORDERS["random-24"] = [rnd.choice([C,S2,S3,S16,S4,(0,0),(1,3)]) for _ in range(24)]
def kernels_compiled():
    out = {}
    for name in ("cross_entropy_fwd_kernel", "cross_entropy_bwd_kernel"):
        k = getattr(ce, name)
        while not hasattr(k, "device_caches") and hasattr(k, "fn"): k = k.fn
        try: out[name] = sum(len(v[0]) for v in k.device_caches.values())
        except Exception as e: out[name] = f"n/a ({type(e).__name__})"
    return out
@pytest.mark.parametrize("order", list(ORDERS), ids=list(ORDERS))
@pytest.mark.parametrize("xdt", ["fp16", "fp32"])
def test_alternation(order, xdt):
    bad = []
    for i, (st, off) in enumerate(ORDERS[order]):
        fwd, bwd, fe, be = run_case(n=129, V=1025, xdt=DT[xdt], ldt=torch.long, stride=st, offset=off, seed=100 + i)
        if not (fwd and bwd): bad.append((i, st, off, fwd, bwd, fe, be))
    print("compiled variants:", kernels_compiled())
    assert not bad, f"wrong results at steps (idx, stride, offset, fwd_ok, bwd_ok, fwd_err, bwd_err): {bad}"
def test_zz_variant_count():
    print("FINAL compiled variants in this process:", kernels_compiled())
