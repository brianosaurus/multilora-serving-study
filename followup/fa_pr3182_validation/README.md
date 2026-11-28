# PR #3182 validation on RTX PRO 6000 Blackwell (2026-11-28)

Nothing here has been pushed or posted. AI assistance (Claude Code) ran and analyzed this under the author's review.

- `harness/`: `fa_noext.py` (import shim + per-test memory recorder), `reproduce_issue.py`, `test_tensor_layouts.py`, `test_specialization.py`, `bench_proc.py`, `bench_driver.py`, `analyze_bench.py`, `run_suite.sh`.
- `raw/`: pytest logs and junit XML (stock, patched, alt, main+PR), per-test memory CSVs, benchmark JSON (52 blocks, raw per-iteration times), driver log, repro output, environment records (`raw/env/`), PR diff, `.contiguous()` alternative diff.
- `BENCHMARK_RAW_TABLE.txt`: full contiguous and strided tables.
- `REVIEW_SUMMARY.md`, `DRAFT_COMMENT.md` (not posted).

## Method
Trees (all from base `698323347caa34174154bd39ca8d442ba4b9bf64` unless noted): `stock`, `stock2` (identical copy, unchanged-code control), `patched` (PR head `c065a938b17714261c2c4524cf1b0097170cf60b`), `alt` (base plus `labels = labels.contiguous()` at the top of `CrossEntropyLoss.forward`, local branch `alt-contiguous` on the pod, not pushed). Each run is a separate process with its own Triton cache. The compiled FlashAttention extension is not built: `fa_noext.py` registers the package from the chosen tree without running `flash_attn/__init__.py`; the module path is asserted on every run.
Benchmark: CUDA events, 3 s clock warm-up, 30 warm-up and 100 timed iterations per (config, mode), 10 blocks (contiguous) or 6 blocks (strided) in randomized order per block, one process on the GPU at a time. Reported value: median over blocks of the per-block median ratio.

## Environment
Host `1730effaa88e` (RunPod). NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB, SM120, driver 595.112.07, CUDA toolkit 13.2, torch 2.8.0+cu129, Triton 3.4.0, Python 3.12.3, pytest 9.2.0. AMD EPYC 9535 (2 sockets, 256 CPUs, container quota about 27), 1.1 TiB RAM visible. Load average 9 to 14 from other tenants (shared host). GPU clocks ramp from 180 MHz idle to about 2420 MHz under load; no throttle reasons set at the sampled points. Full records in `raw/env/`.
