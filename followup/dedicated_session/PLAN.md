# Dedicated-machine session (plan written 2026-11-24 about 09:00 UTC local, before any run; no run has been made)

Author's decision, 2026-11-24: every timing run moves to a dedicated machine, because the pod host is not quiet (external CPU 6 to 17 cores). One script, `run_session.sh`, runs all of them in one session with the checks of `run_rerun.sh`. The machine is not chosen yet; `--host` names it and the script refuses to start on any other hostname. Same GPU type as the study (RTX 4090) is assumed; if it differs, every number is labeled with the GPU and not compared with the study's without saying so. Exploratory, separate from the pre-registered study.

## What runs (order fixed; seed 20261124 unless the author sets another; the seed shuffles only the order of server sessions inside a block and the cell order in B2)

- **B1: 3a repeat.** SGLang 0.5.23, radix cache on, the cold and warm mixed probe passes of `followup/prefix_routing/PLAN.md`, same prompts, references and rule, with the quiet checks around them. Output tag `passes_quiet_repeat` so the flagged data is not overwritten. If it is clean and routes 64 of 64 in both passes, the flag on the SGLang part of the 3a results is cleared (and only that part; the vLLM fallback comparison is not repeated unless the author asks). If it routes fewer than 64 of 64, that is a finding: stop, reproduce, log in `docs/BUGS.md`.
- **B2: 4a, kernel rerun.** `followup/kernel_rerun/run_rerun.sh` with its own checks, old and new rebuilt binaries and the byte-identical old copy, six block orderings, Amendment 2 below.
- **B3: 4b, cliff rerun with the 0.02 adapters.** vLLM 0.34.0 and SGLang 0.5.23; 32 adapters at B scale 0.02, seeds 0 to 31 (`scripts/make_adapters.py`, checksums of seeds 0 to 3 equal the routing sweep's); 16 and 32 slots; Zipf and uniform popularity; rates 10, 20, 40 rps; 3 repeats; the study's open-loop harness (`scripts/loadgen.py`: 50 warm-up and 500 measured requests, 256 input tokens, 128 output tokens, temperature 0) and its pass rule (p95 TTFT at most 500 ms, median of the 3 repeats). Output length control: for vLLM and SGLang the harness sends `ignore_eos` true and `max_tokens` 128, and the script flags any run in which a measured request did not report exactly 128 output chunks (`LENGTH`). Four server sessions (engine by slots), order shuffled by the seed; each session has two warm-up runs, a reference run at its start and end (rate 20, base model, 200 requests), and is marked DRIFT if the reference median end-to-end latency changes by more than 5%.
- **B4: 3b, SGLang main against v0.5.23.** The study's adapters (B scale 0.05, seeds 0 to 31), 16 slots, defaults and `--lora-drain-wait-threshold 1.0`, for v0.5.23 and for main (the commit recorded in `versions.txt`, installed on the pod as Item A of `followup/pod_items/PLAN.md` and copied or rebuilt on the dedicated machine at that same commit); Zipf at 20, 40, 80 rps and uniform at 20, 40 rps; 3 repeats; plus 32 slots at Zipf 40 rps for the defaults of both builds. Same harness, pass rule and drift check.
- **B5: 3c, 128 adapters on 64 slots.** Study adapters (scale 0.05, seeds 0 to 127), Zipf, 20 and 40 rps, 3 repeats, vLLM `--max-loras 64` and SGLang `--max-loras-per-batch 64`. A server that fails to start (memory) is reported as such.

Estimated length with the quiet waits: B1 10 minutes, B2 about 150 minutes, B3 about 70 minutes, B4 about 130 minutes, B5 about 25 minutes, about 6.5 hours.

## Checks (as `run_rerun.sh`, Amendment 1 of the kernel plan)

Before the first run: hostname equals `--host`; GPU idle; 1-minute load average at most 2; external CPU at most 1 core (median over 10 s). Before every run: external CPU at most 1 core over 6 s (waits up to 120 s, then runs and flags). During every run: any sample above 2 external cores flags it `LOAD_ABOVE_LIMIT`. A run flagged is reported and excluded from every verdict below; a verdict that needs a flagged or missing run is printed as "not evaluated". A session with reference drift above 5% is marked DRIFT and its verdicts are not evaluated. Software versions and binary checksums are written to `versions.txt` at the start.

## Definitions, fixed before any run (all on p95 TTFT, median of 3 repeats, "min-max" the range of the 3)

Reference values are the study's main grid and follow-up data in `followup/sglang_issues/FACTS.md` (A1 to A5).

- **"The cliff reproduces" (4b), per engine and popularity at 16 slots:** the median p95 TTFT is at most 500 ms at the rate that passed in the main grid (Zipf 20 rps, uniform 10 rps, both engines) and over 500 ms at the next rate up (Zipf 40 rps, uniform 20 rps). Else "does not reproduce". The study's numbers at scale 0.05: vLLM Zipf 50 then 1456 ms, uniform 51 then 684 ms; SGLang Zipf 84 then 12142 ms, uniform 81 then 755 ms.
- **"The slot fix reproduces" (4b), per engine:** at 32 slots, Zipf, 40 rps, the median p95 TTFT is at most 500 ms. (Main data: vLLM 142 ms, SGLang 101 ms at scale 0.05.) Uniform popularity at 32 slots has no prior value in the data and is reported without a verdict.
- **3b, per cell (workload, rate, variant), main against v0.5.23 from the same session:**
  - the baseline must show the cliff first: v0.5.23 defaults, 16 slots, Zipf 40 rps, median over 500 ms (the study measured 12031 ms); if not, the comparison is not interpreted;
  - **improved:** main's median is at most 80% of v0.5.23's and main's max is below v0.5.23's min;
  - **worse:** main's median is at least 120% of v0.5.23's and main's min is above v0.5.23's max;
  - **unchanged:** neither. A cell where main's median is at most 500 ms and v0.5.23's is over 500 ms is additionally noted as "main passes the bar".
  - Drain-threshold variants are compared like for like (v0.5.23 with 1.0 against main with 1.0). The reference values in FACTS A4 (defaults 12031 ms at 40 rps, 9672 ms at 80, drain 1.0: 10111 ms and 9279 ms) are context only; the same-session v0.5.23 run is the comparator.
- **"The cliff moved" (3c), per engine:** with 128 adapters on 64 slots, Zipf, the median p95 TTFT at 40 rps is at most 500 ms (the same adapters on 32 slots gave vLLM 3011 ms and SGLang 11662 ms at 40 rps, FACTS A5). 20 rps is reported as context.
- Any statement beyond these is not a result of this plan. No cell is called significant; ranges are shown instead.

## Amendment 2 to `followup/kernel_rerun/PLAN.md` (4a, shortened design)

The design of the benchmark is unchanged (6 blocks, 3 binaries in 6 balanced orderings, 150 requests per run, the establishment rule). One server start per binary per block with 8 adapters loaded; all 8 cells (0, 1, 4, 8 active adapters, varying adapters per request, at concurrency 8 and 32) run against that server; the order of those cells is shuffled within each server session with a recorded seed (`--seed`, seed + 10 times the block + the position), so cell order is not confounded with elapsed time. The check rules of Amendment 1 apply. Written here before any rerun on the dedicated machine; `attempt1_INVALID` data stays unused.

## Needs on the dedicated machine (checked by `preflight.sh`; the session refuses to start if any is missing)

The repo at `/workspace/study`; Python environments `/workspace/envs/{vllm,sglang,sglang_main}` (versions as `followup/pod_runpod2/SETUP.md`; main at the recorded commit); the model in `/dev/shm/models/Qwen2.5-1.5B-Instruct`; 128 adapters at scale 0.05 in `/dev/shm/adapters`; 32 adapters at scale 0.02 in `/dev/shm/adapters_s0.02_32` (the 3a repeat also needs the first 4 of these as its adapters, so `A_DIR` points there); the two mistral.rs binaries `/workspace/bin/mistralrs_old` and `mistralrs_new` (from Item D); `refs_prefix.json` in `followup/prefix_routing/results/raw/`.

## Not covered

B1 to B5 give measurements on one machine, one GPU, one model, rank 16; the dedicated machine is assumed quieter, not proven so (the checks flag it when it is not). Results will be reported with the flags, and nothing is claimed beyond the definitions above.
