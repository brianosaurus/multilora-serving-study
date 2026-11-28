# Draft for PR #3182 (NOT POSTED)

Follow-up on an RTX PRO 6000 Blackwell (96 GB, SM120, torch 2.8.0+cu129, Triton 3.4.0). Base `6983233`, PR head `c065a93`.

- **Memory-limited cases:** the 54 cases that ran out of memory on the 4090 now run. `test_cross_entropy.py` collects 384 cases on base: 240 pass, 144 skip by design (`precompute_lse` needs `logit_scale=1.0` and `smoothing=0.0`), 0 fail, 0 OOM. With the PR: 243 pass (the 3 new strided tests), 0 fail. The new test fails on base (3 of 3).
- **Layouts:** 191 extra cases (strides 2 to 16, storage offsets, zero-stride labels, sizes near block boundaries, smoothing, logit scale, z-loss, `ignore_index`, int32 labels), forward and backward against an fp32 PyTorch reference. 187 pass with the PR; base fails the 122 aligned strided ones. The other 4 (`vocab_size=1`, Triton compile error) fail on base too. Alternating layouts in one process stayed correct.
- **Contiguous labels:** 10 alternating blocks plus an unchanged-code control. Ratios to base are 1.000 from 8192 rows up. Small launch-bound shapes are noisy, and the control varies as much as the patch. No meaningful regression was observed under these workloads.
- **Alternative:** a wrapper-level `.contiguous()` also passes, within noise for large batches (about 4 us slower on a 512-row forward). Either works for me.

Limits: one GPU, one torch/Triton version, no tensor-parallel run, CUDA extension not built. Applies cleanly on current main. Raw logs and scripts: <link once pushed>. AI assistance (Claude) helped run and analyze this.

---

# Draft maintainer request (NOT POSTED; second ping to the same reviewer)

@Johnsonms the validation above covers the cases that did not fit on a 4090. The change is +36/-2 in `cross_entropy.py`. Review whenever you have time.
