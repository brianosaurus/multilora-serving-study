# Build record for the kernel rerun (Part B1 and B3)

Recorded 2026-11-23 on the follow-up pod. The two binaries below are the ones the quiet-machine benchmark (`run_rerun.sh`) will use. mistral.rs here is upstream `master`, the 0.10.0 development line that PR #2571 is based on. It is not the study's pinned v0.9.5.

## What was built

- Base commit: `21da642fdd2210c30ca4028e1eabdd28e9c602fd` ("feat(core): maintainability refactor (#2561)").
- **Old tree:** a clean checkout of the base commit. **New tree:** the same checkout with the PR's commit applied from the formatted patch (4 files changed, 570 insertions, 12 deletions: `mistralrs-quant/kernels/lora/dynamic_lora.cu`, `src/lora/dynamic/cuda.rs`, `cuda_ffi.rs`, `execution.rs`). A recursive diff of the two trees (excluding `.git`) shows exactly those 4 files.
- Command, identical for both: `cargo build --release -p mistralrs-cli --features "cuda flash-attn"`, each in its own empty target directory.
- Environment, identical: `PATH` with `/usr/local/cuda/bin`, `CUDA_HOME=/usr/local/cuda`, `CUDA_COMPUTE_CAP=89`, `CARGO_BUILD_JOBS=8`, `CUDAFORGE_THREADS=4`, a local `TMPDIR`, `RUSTFLAGS` unset. Toolchain: rustc 1.100.0 (0ef1bc386 2026-11-09), cargo 1.100.0 (3acc79c0c 2026-11-05), nvcc cuda_12.8.r12.8 (build 35583870_0). GPU: RTX 4090 (sm_89).

| | Old tree | New tree (PR commit) |
|---|---|---|
| Build time | 60 min 22 s | 53 min 30 s |
| Binary size | 549,206,280 bytes | 549,255,576 bytes |
| SHA-256 | `56fbcbc6dab8ba581a745ecc6f50f94faf0b372f8f65902f64ac7a01f6fe9d9f` | `d0b658c25946c4f6d4840a4bc2287a768dba7d5ecac1aaef1d15b66c045ec5de` |

## The earlier benchmark builds against these

The earlier first and balanced benchmarks (`results/raw/kernel_followup/`) used two other binaries, built from the same base commit.

| | Earlier builds | Rebuilds |
|---|---|---|
| Base commit | `21da642` | `21da642` |
| Cargo features | `cuda` | `cuda flash-attn` |
| Flash-attention kernels in the binary | none (0 kernel markers) | present (30,600 markers) |
| Binary size | 301,156,192 (old) and 301,206,896 (new) | 549,206,280 and 549,255,576 |
| `CARGO_BUILD_JOBS` / `CUDAFORGE_THREADS` | 6 / 2 | 8 / 4 |
| Target directory | old: empty; new: the old's directory, built incrementally after the patch | each empty |
| Patch applied to the new tree | the unformatted version, before `cargo fmt` and `clang-format` | the formatted version (formatting only, same content otherwise) |
| rustc / cargo / nvcc / `CUDA_COMPUTE_CAP` / `RUSTFLAGS` | 1.100.0 / 1.100.0 / 12.8 / 89 / unset | the same |
| Second old copy (`mrs-base2`) | byte-identical to the old build (same SHA-256) | not used here; the benchmark script makes its own copy |

**Did the earlier builds differ in anything besides the PR commit?** Yes, in four configuration points, none of which would explain a speedup on code the PR did not touch: (1) no flash-attn, which is also unlike the study's own binary; (2) the parallelism settings, which change build speed and not the generated code; (3) the new build was incremental on top of the old build's target directory; (4) the new build's source was the pre-formatting version of the patch. I did not find a difference in toolchain, base commit, CUDA architecture or `RUSTFLAGS`. In the first benchmark the byte-identical copy of the old build showed a same-code difference of the same size as the new build's on the 1-adapter cell (13.5% against 19%), so that effect is not evidence of a build difference.

## Checks on the rebuilt trees

- `cargo fmt --all -- --check`: exit 0 in both trees.
- LoRA unit tests on the new tree (`cargo test --release -p mistralrs-quant --features cuda --lib lora::dynamic`): **81 passed, 0 failed**, including the grouped-versus-loop tests.
- `cargo clippy -D warnings` on the rebuilt trees: **not yet run on these rebuilds** (a full recompile; waiting on disk and the v0.9.5 build). The same source passed clippy earlier in this follow-up on a `cuda` build.
- End-to-end, 128 completions (16 prompts by 8 adapters at B scale 0.05, temperature 0, 8 distinct adapters in flight, `results/raw/b3/`), old against new, 6 launches of the old build and 5 of the new build:
  - **The criterion "match bit for bit" is not met as written:** 20 of 30 old-versus-new launch pairs were fully identical, 10 were not (minimum 120 of 128 identical).
  - All differences are within the old build's own launch-to-launch variation. 120 completions were identical in every old launch and are identical to the new build's output. 8 completions vary across old launches (the first prompt, "The capital of France is", with each of the 8 adapters, so the first request of each wave after a fresh start); the new build gave one output for each of these in all 5 launches, and that output is one of the values the old build produced (8 of 8).
  - The old build is the less deterministic: old against itself, 15 pairs, only 6 fully identical; new against itself, 10 pairs, all 10 fully identical. I did not establish why. A likely explanation is that the first wave of requests after a restart is batched differently, and the old build's different timing lets the batch composition vary, but I did not test that.
