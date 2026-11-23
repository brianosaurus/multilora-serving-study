# Follow-up B: grouped-kernel rerun on a quiet machine (plan, fixed before any run)

Exploratory follow-up after the study's hard stop, separate from the pre-registered study. The first benchmark (`results/raw/kernel_followup/`) ran on a shared, loaded host (load average 25 to 250). Two questions follow from it: why the new build looked faster on code paths it did not change, and whether the speedup holds on a quiet machine. Changes after the first run are logged in `docs/CHANGELOG.md` with the reason.

## What the first benchmark showed, for reference

At 32 concurrent requests with 1 adapter (identical code in both builds) the new build was faster by 19% in the median and in 5 of 6 blocks. The old-build copy (same bytes as the old build) was also faster than the old build by 13.5% in the median on that cell. So on that cell the same-code effect was as large as the claimed one. This plan measures the spread properly instead of arguing about it.

## B1: explain the same-code difference before rerunning

1. Record, for the two existing binaries (`mrs-base`, `mrs-grouped`) and the copy (`mrs-base2`): size, SHA-256, `file` and `readelf` build attributes, linked CUDA libraries, embedded version strings, and whether `mrs-base` and `mrs-base2` are byte-identical.
2. Diff what could differ between the two builds: base commit, cargo features, profile, `RUSTFLAGS`, `CUDA_COMPUTE_CAP`, nvcc and rustc versions, whether flash-attn was compiled in (neither existing build had it; the study's binary did), the order the builds ran in, and the target directory state (the second build was incremental).
3. Rebuild both from clean trees with identical flags, from the same base commit (`21da642`, the commit the PR is based on), differing only by the PR's commit: `cargo build --release -p mistralrs-cli --features "cuda flash-attn"`, separate empty target directories, same environment. Record both binaries' size and hash. Report what differed, if anything.

## B2: benchmark on an idle machine

- Machine: the study's RTX 4090 if available. The study's pod no longer exists. The condition is a load average under 2 throughout, recorded for every run along with GPU clocks, temperature and power. Runs on a machine that cannot meet it are not reported as the rerun (they are reported as "not run, no quiet machine"). Nothing else may use the GPU, and the GPU must be idle before each run.
- Design, as before: three binaries (old build, new build, a second copy of the old build) in all 6 orderings, 150 requests per run, closed loop, a warm-up per server. Concurrency 8 and 32. Adapters 0, 1, 4 and 8; 0 and 1 adapters are unchanged-code controls. Qwen2.5-1.5B-Instruct, the study's adapters, the flash-attn build (the study's binary was built `cuda flash-attn`).
- Per cell: median time per output token and throughput for each binary, per-block changes, in how many blocks the new build was faster, and the control spread.
- Rule, fixed here: the kernel's effect is established at a concurrency only if, at the 8-adapter cell (and separately at the 4-adapter cell), (a) the new build is faster in all 6 blocks and (b) the median change is larger in magnitude than the largest median change seen at the same concurrency in the unchanged-code cells (0 and 1 adapters, new build against old build) and in the old-versus-old-copy control. Concurrency 8 and 32 are stated separately. Otherwise the result is "not established".

## B3: correctness on the rebuilt binaries

- The LoRA unit tests on the GPU (including the two grouped-versus-loop tests).
- The 128 end-to-end completions (16 prompts, 8 adapters, temperature 0), old build against new build, with the study's adapters: they must match bit for bit. Any difference is reported with its request and, if it is a near-tie, the logprobs.

## Gate B

The build diff, the results table with both criteria, the correctness results, and raw file paths. If no quiet machine is available, B2 is reported as not run, with the load averages seen, and B1 and B3 are reported as done.

## Limits stated in advance

One model, one GPU type, rank 16, up to 8 adapters. A quiet machine removes the host-load noise; it does not remove run-to-run variation inside the machine, which the old-versus-old control measures.

## Amendment 1: what "quiet" means (author approved, 2026-11-23 about 07:58 UTC pod clock, before any rerun)

Why: a measurement on the idle pod (`results/raw/selfload/`) showed one 32-concurrent, 8-adapter run takes the 1-minute load average from 1.0 to 2.8 by itself (the host CPU was 3.7 cores busy, all of it the benchmark's own server and load generator). "Load average under 2 throughout" therefore cannot hold for any machine while the benchmark runs, and the script would flag every run.

Amendment (the author's wording of the idea, with one refinement stated here):
- The condition is on load from other tenants, not the benchmark's own. External CPU use = host-wide busy cores (from `/proc/stat`) minus the cores used by processes in this container (summed from `/proc/<pid>/stat`), sampled every 2 s.
- Before the first run: the 1-minute load average must be at most 2 and external CPU use at most 1 core (median over 10 s). The load average is checked only here, because later in the benchmark it includes the benchmark's own earlier runs.
- Before each later run: external CPU use at most 1 core over the preceding 6 s.
- During each run: any sample with external use above 2 cores marks the run `LOAD_ABOVE_LIMIT`. The load average is still recorded at every sample, for information only.
- A rerun with any flagged run is reported as invalid, as before. The design, the 6 blocks and the establishment rule are unchanged.
- Known bias, stated in advance: if a container process exits between two samples its CPU time disappears from the "own" sum, which makes external use look larger, so the check errs toward flagging.
