# Review summary (PR #3182, RTX PRO 6000 Blackwell)

## Results
| Test category | Collected | Passed | Failed | Skipped | OOM |
|---|---:|---:|---:|---:|---:|
| Existing `test_cross_entropy.py`, base | 384 | 240 | 0 | 144 (by design) | 0 |
| Existing file with PR (incl. 3 new strided tests) | 387 | 243 | 0 | 144 | 0 |
| Layout cases (`test_tensor_layouts.py`), PR | 191 | 187 | 4 (pre-existing) | 0 | 0 |
| Layout cases, base | 191 | 65 | 126 | 0 | 0 |
| Alternation (`test_specialization.py`), PR / base | 15 / 15 | 15 / 1 | 0 / 14 | 0 | 0 |
| Everything on current main (`4b6dee0c`) + PR | 593 | 445 | 4 (pre-existing) | 144 | 0 |

- The 4 failures are `vocab_size=1`: Triton compile error, identical on base, PR, `.contiguous()` alternative and main. Unrelated to the patch.
- Base fails all 122 aligned strided cases in both forward and backward. Strided labels at unaligned addresses already go through the wrapper's `F.pad` copy, so base is correct for them and the new stride argument is not exercised.
- The 54 cases that exceeded 24 GB peak at 43.05 GiB (`vocab_size=128256`); they pass here.
- The issue repro fails on base (loss error 3.0, gradient error 1.0) and passes with the PR (errors 1.5e-8 and 7.5e-9; the PR text says 0.0).

## Performance, contiguous labels (16 configs: fp16/bf16, V 32000/128256, 16 to 32768 rows)
- n >= 8192: ratio to base 1.000 for stock-copy control, PR and `.contiguous()` in every block.
- n <= 512 (launch bound, 35 to 150 us): medians within 0.977 to 1.024 of base for the PR, 0.981 to 1.016 for the control; per-block spread is wide for all variants (0.36 to 2.8), so these shapes cannot resolve differences below a few percent.
- Statement supported: no meaningful regression observed under the tested contiguous-label workloads. This is not a zero-overhead claim.

## PR vs `.contiguous()` (strided labels, strides 2 and 16, fp16)
- Correctness: both pass every layout and alternation test.
- Large batches (>= 8192 rows): alt/PR 0.999 to 1.010 (fwd up to +1%), fwd+bwd 0.999 to 1.001.
- 512 rows: alt is 7 to 12% slower in forward (about 3 to 4 us, one extra copy kernel), noisy in backward.
- Extra memory: alt +0.3 MB at 32768 rows (8 bytes per row); negligible against 2 to 8 GB of gradients.
- Compiled variants in one process: PR 6 per kernel (2 dtypes x stride classes 1 / generic / divisible by 16), base 2. Non-contiguous labels pay one extra JIT compile per new stride class; contiguous labels pay none.
- Complexity: alt is one line; the PR is +6/-2 in the kernel file plus a test. Both are correct. Maintainer preference decides.

## Adversarial review
- Medium: the PR test covers one shape (512x256), stride 2, aligned. The wider layout tests live only in this folder. Offer to add a parametrized stride/offset case if the maintainer wants broader coverage.
- Low: the PR text says gradient error 0.0 on the 4090; here it is 7.5e-9 (hardware and Triton difference). Harmless.
- Low: `int32` labels bypass the wrapper's `F.pad` alignment workaround; tested correct, strided and mixed with `ignore_index`.
- Info: wrapper `F.pad` workaround hides the bug for unaligned strided starts; the issue's repro needs an aligned allocation.
- No bug found in the PR. Backward and forward use the same stride expression and cast `row_idx`, which avoids the Triton constant-specialization crash for stride 1.

## Not covered
Tensor-parallel (`process_group`) path (single GPU); other torch/Triton versions (issue reported torch 2.14.1, Triton 3.8.0 in interpreter mode); SM89 here (4090 numbers are from the PR); CUDA extension not built; per-block variance on the shared host for shapes under 1024 rows.

## Recommendation
Leave PR #3182 unchanged. Offer the `.contiguous()` comparison in the comment so the maintainer can choose.
