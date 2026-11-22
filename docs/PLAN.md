# Plan

Status: decisions fixed, no measurement run yet. The answers below were fixed on 2026-11-21, before any experiment was run, and are subject to the feasibility checks listed after them. Any change after the first run is recorded in `docs/CHANGELOG.md` with the reason, and this repository's history is not rewritten.

## Decisions

The Day plan below overrides these rows where they conflict: adapter counts, resident slots, traffic (the rate sweep and request counts), repeats, and mistral.rs.

| Item | Decision |
|---|---|
| Base model | Qwen2.5-1.5B-Instruct for the main sweep. A Llama-class model only if an engine cannot do LoRA on Qwen. |
| Adapters | Synthetic, built with PEFT: rank 16, all linear layers (q, k, v, o, gate, up, down), a distinct seed per adapter, B matrices set to nonzero. A few real adapters from the Hugging Face Hub may be used as a sanity check only. |
| Adapter counts | Main sweep: 1, 8, 32, 128 total adapters. Replication grid: the published one (10, 50, 100, 500), or the closest feasible. Adapter files are kept in `/dev/shm` (RAM-backed, data only), not on the small root disk. |
| Resident slots | Vary `max_loras` (SGLang: `--max-loras-per-batch`) over 4, 16, and 32. |
| Traffic | Open-loop Poisson arrivals at a rate sweep. Adapter popularity: uniform, and Zipf with exponent 1.0. Fixed 256-token inputs and 128-token outputs, plus one variable-length run as a robustness check. 100 warm-up and 1,000 measured requests per point. |
| TTFT limit | p95 TTFT of 500 ms or less defines peak throughput. p99 is reported as well. |
| Metrics | TTFT p50/p95/p99, TPOT, end-to-end p50/p95, output tokens per second, requests per second, peak GPU memory, error and timeout counts. |
| Repeats | 5 runs per configuration, order randomized across engines. Median plus min-max reported. A difference under 10%, or inside the run-to-run spread, is reported as inconclusive. |
| Replication A | SGLang docs claim: `csgmv` gives 20% to 80% lower latency than the Triton backend. Held if median p95 latency is at least 20% lower with `csgmv` at concurrency 32 and above, with non-overlapping ranges. Partly held at 5% to 20%, or at only some points. Not held otherwise. |
| Replication B | Public vLLM sweep: `max_loras=16` is best for 10 to 500 adapters under skewed traffic. Held if 16 is the best, or within noise of the best, at every adapter count tested. Not held if another setting beats it beyond the run-to-run spread at two or more adapter counts. |
| Replication method | Replications follow the original claim's method as closely as it is documented, including closed-loop concurrency where the claim is stated by concurrency. The main comparison is open-loop. |
| Engine versions | The latest stable release of each engine at the start of the study. No upgrades mid-study. If a major release ships, the key results are rerun once and both are reported. |
| mistral.rs | In the outcome-level comparison, with no pass/fail replication thresholds, labeled as not like-for-like on the knob level. |
| GPU hygiene | Confirm the GPU is idle before each run, record clocks and temperature, and discard runs with thermal throttling. |
| Hardware | Every engine, baseline, and replication runs on the same single RTX 4090 in the same machine, one engine process at a time, with nothing else on the GPU. The same checkpoint and adapter files are used by every engine. Because the engines share one GPU over weeks, a fixed reference run (same engine, same configuration) is repeated at the start and end of every session; a session whose reference differs by more than 5% is discarded and rerun. |
| Storage | Engine installs, virtual environments, downloaded checkpoints, and results may live on network storage (the machine's existing network-mounted volume at `/workspace`, measured at about 47 GiB usable, with about 660 MB/s sequential reads and about 17 times slower small-file reads than local disk). Nothing read during a timed run comes from it: the model checkpoint and all adapter files are copied to RAM-backed storage (`/dev/shm`) before an engine starts, and compiler and JIT caches (Triton, FlashInfer, torch compile, Hugging Face) are kept on local disk. Warm-up requests and the first requests of every run are excluded from results, so load time cannot leak into them. The storage layout and the measured read speeds of each location (large sequential reads and small-file reads) are recorded with the results. |
| Relative metric | Two levels, both reported. Level 1, within each engine: adapter versus no adapter. The no-adapter baseline (B0) is the same engine, base model, and traffic, started with LoRA disabled. Where the engine allows it, a second baseline (B1) with LoRA enabled but requests selecting no adapter is also reported, to separate the cost of enabling LoRA from the cost of using adapters. The overhead factor is adapter divided by baseline for latencies, and baseline divided by adapter for throughput, so 1.0 means no cost and larger means worse. Peak throughput under the TTFT limit is compared the same way. The range comes from the 5 repeats: from (min adapter / max baseline) to (max adapter / min baseline). Level 2, across engines: each engine's overhead factors are compared with the other engines', and the absolute results on the identical workload are shown beside them. Neither level is reported alone. Engines are not ranked when their ranges overlap. |

## Day plan (operative)

Constraint: the whole study, from first setup to published write-up, runs in under one day (24 hours wall-clock). This was fixed on 2026-11-21, before any run, because of the author's time constraint. Where it conflicts with the Decisions table, this section wins. The full grid is not planned.

| Item | Day plan |
|---|---|
| Workloads per engine | B0 (LoRA disabled), B1 (LoRA enabled, no adapter selected, where the engine allows it), 1 adapter, 32 adapters with Zipf popularity, 32 adapters with uniform popularity |
| Adapter counts | 1 and 32 in the main grid. Replication B uses 10 and 100. |
| Resident slots | Fixed at 16 in the main grid. The 4/16/32 sweep exists only in replication B. |
| Rate ladder | 5, 10, 20, 40, 80 requests per second. A step is a violation if the median p95 TTFT of its runs exceeds 500 ms. The ladder stops after two consecutive violations. |
| Repeats | 3 per step. If time remains, the highest passing step and the first violating step of each workload are extended to 5. |
| Requests per run | 50 warm-up and 500 measured |
| Dropped | The variable-length run, the 500-adapter replication grid, and the real-adapter sanity check |
| mistral.rs | A reduced set: B0, 1 adapter, and 32 adapters with Zipf popularity (B1 only if it means something for that engine) |
| Replication A | `csgmv` versus Triton in SGLang, closed-loop concurrency 8, 32, and 128, 3 repeats |
| Replication B | vLLM `max_loras` of 4, 16, and 32, at 10 and 100 adapters, Zipf popularity, 3 repeats |
| Tuning | 3 trials per engine on a separate tuning workload (different seed and prompts from the measured workloads). The settings list is still open; see below. |

Clock: hour 0 is 2026-11-21 23:25:58 UTC, the moment the engine installs began, as the author specified. The hard stop is 2026-11-22 23:25:58 UTC.

Prerequisites: exclusive use of the machine for the day, the 50 GB network volume attached, nothing else running on the GPU, and the machine's identity verified before any action.

Schedule, as estimates until runs are timed:

| Phase | Target | Gate at the end |
|---|---|---|
| Installs and each engine serving the base model | hours 0 to 2 | Gate 1: all engines answer a base-model request |
| Correctness checks (adapter applied, against a PEFT reference) | hours 2 to 5 | Gate 2: all engines pass |
| Harness, tuning, and smoke runs | hours 5 to 8 | Gate 3: the harness produces valid smoke results; the tuning protocol is committed |
| Measurement | hours 8 to 15 | |
| Analysis and write-up | hours 15 to 18 | |
| Buffer | hours 18 to 24 | Hard stop at hour 24 |

If a gate is missed, work is cut in this order: mistral.rs, then the 32-adapter uniform workload, then B1, then replication B. Never cut: the correctness check, B0, the 32-adapter Zipf workload, and replication A. The repeats do not drop below 3. At hour 24 everything stops; completed items are reported, and anything unfinished is reported as not done.

What this costs in conclusions: only two adapter counts, so there is no scaling curve; ranges from 3 repeats are wide, so more results will be reported as inconclusive; the study is a single-day measurement and is labeled as one.

## Correctness check

Criteria fixed on 2026-11-22, approved by the author, before any engine was tested with an adapter. The check must establish that each engine applies the right adapter, including when requests for different adapters share a batch.

Setup: 4 adapters (seeds 0 to 3), rank 16, alpha 32, all linear layers, float16. 16 fixed prompts (`configs/check_prompts.json`), greedy decoding, 32 new tokens. The reference is transformers plus PEFT with unmerged adapters, in bf16, one prompt at a time. The scale of B is chosen by calibrating against the reference only.

Criteria:

1. Calibration (reference only): for every adapter, the output differs from the base model's within the first 8 tokens on at least 14 of 16 prompts. The candidate scales for B are 0.01, 0.02, and 0.05, and the smallest that passes is used. If none passes, work stops and the failure is reported; the criterion is not changed.
2. Adapter identity, serial: for each engine, adapter, and prompt, the engine's output is compared with the reference outputs of the base and of all 4 adapters. It passes if its longest shared prefix is with the correct reference; a tie is a failure. At least 90% of prompts must pass, per engine.
3. Adapter identity, mixed batch: the same test with the requests of all 4 adapters sent concurrently (64 requests). At least 90% must pass, and each request must be matched to its own adapter's reference.
4. Logprob closeness for the first token is reported where an engine exposes it. It is not a pass/fail gate.

If an engine fails, the cause is investigated first (adapter dtype, target modules, loader limits). If it cannot be fixed, that engine is excluded from adapter timings and the failure is reported as a finding.

Why prefix matching: engines that are all correct still diverge after a few tokens at temperature 0 because of bf16 near-ties (the Gate 1 replies agreed on the first two tokens and then differed). Exact full-text equality would fail correct engines; identifying the right adapter by longest prefix tolerates that noise.

Judgment calls, fixed now so they cannot move later: the 90% threshold, the 8-token calibration window, the 32-token length, and the candidate scales for B.

How each engine selects an adapter per request differs (vLLM `--lora-modules`, SGLang `--lora-paths`, mistral.rs `--lora` aliases) and is recorded per engine in `configs/`.

### Amendment (post-hoc), 2026-11-22, approved by the author

Criteria 2 and 3 above were applied exactly as written and every engine failed, and so did a known-correct implementation (the noise floor, 43.8%): the criterion has no power with these adapters. This amendment was made after seeing the engine results, so it is weaker evidence than a criterion fixed beforehand, and the original results stay in the changelog and in `results/raw/correctness/`.

Replacement for criteria 2 and 3, fixed before any result with the new metric exists:

- Metric: for each request, the engine returns its top-20 log-probabilities for the first generated token. The reference provides the top-20 for the base model and for each adapter (transformers plus PEFT, unmerged, bf16). The distance to a variant is the mean absolute difference in log-probability over the reference's top 10 tokens for that variant; a token the engine does not return counts as the engine's lowest returned log-probability. The nearest variant is the one with the smallest distance, and a request passes if that is the adapter it asked for.
- Gate: at least 90% of requests pass, per engine, in serial mode and in mixed mode (the same 64 requests as before).
- Validity of the metric: the noise-floor implementation (adapter merged into the weights, bf16) must pass at least 90% on this metric. If it does not, the metric is declared invalid and no engine is certified by it.
- An engine that does not return log-probabilities cannot be checked with this metric; that is reported as "not checkable", not as a pass.
- Reported with the result: the mean distance to the correct variant, and the 64-request accuracy of the text-prefix criterion (as written above) for comparison.

## Exploratory bug hunting

A secondary goal of the study is to find reproducible, fixable bugs in the engines. Anything that looks like a defect is logged in `docs/BUGS.md` with what is verified and what is not. The extra probes listed there run only when no measurement is in progress, since the GPU is used by one process at a time. They are exploratory, they do not change any pre-registered criterion, and they are reported separately from the measured results.

## Tuning protocol

Fixed on 2026-11-22 and approved by the author, before any timed run. Every engine gets the same budget from the same documented list, chosen on a workload that is separate from the measured ones.

- Fixed for every engine and not tuned: the model; 85% of GPU memory for vLLM and SGLang (an explicit 8 GiB KV cache for mistral.rs, because its default memory plan returned errors on every request on this GPU); 16 LoRA slots; a context length of 4096 where the loader supports it; the engine's default attention and graph settings.
- Tuning workload: adapters 100 to 107 (8 adapters), Zipf popularity (exponent 1.0), 256 input and 128 output tokens, a seed and prompts different from the measured workloads, rates of 10, 20, and 40 requests per second, 50 warm-up and 300 measured requests per run, one run per rate.
- Three trials per engine from one list:
  - T1: the engine's defaults.
  - T2, throughput-oriented: at most 256 concurrent sequences (vLLM `--max-num-seqs`, SGLang `--max-running-requests`, mistral.rs `--max-seqs`).
  - T3, latency-oriented: a 2048-token prefill budget (vLLM `--max-num-batched-tokens`, SGLang `--chunked-prefill-size`, mistral.rs `--max-num-batched-tokens`).
- Winner rule: the trial with the highest throughput at which p95 TTFT stays at or below 500 ms with no errors. If the best trial is within 5% of T1, T1 wins.
- The winner's settings are then used for every measured workload of that engine. Tuning data is kept in `results/raw/tuning/` and is not part of any reported result.
- Baselines (B0, LoRA disabled) use the same winning settings with LoRA flags removed.

## Still open

Nothing about the design is open. The feasibility checks below remain, and how each engine behaves on the output-length control (`ignore_eos`, which vLLM and SGLang take) is checked at harness time and reported.

## Feasibility checks, not yet done

If any check fails, the affected decision is revisited and the change is logged in `docs/CHANGELOG.md` with the reason.

- Each engine can run LoRA on Qwen2.5-1.5B with all of the target modules above (SGLang, vLLM, and mistral.rs on safetensors weights).
- The replication model: the public vLLM sweep used Llama-3.1-8B, which is gated. If it cannot be used, the closest feasible model is used and the replication is labeled as not identical.
- Whether the 500-adapter replication grid fits on 24 GB; if not, the largest feasible grid is used.
- Engine installs on the GPU machine, and enough disk for models and adapters.
- Network volume capacity: the engine installs, the checkpoint, and the results must fit in about 47 GiB (measured on 2026-11-21 by writing until the quota refused; an earlier write failure on this volume was a full quota, and it was emptied before the study). If the replication model needs more (Llama-3.1-8B is about 16 GB), this decision is revisited.
- Network volume behavior: confirm a real write succeeds (an earlier volume on this kind of machine reported free space but rejected writes with a quota error), and measure its read speeds before relying on it. Install time on network storage is slower for small files, so installs are timed and counted in the schedule.
- Load time does not leak into timed results: compare the first requests of a run with steady state, and confirm adapters and weights are read from RAM-backed storage.
- The 20% and 10% thresholds are judgment calls made before seeing any data; they are fixed now so they cannot be moved afterward.

## Method

Replication first, matched comparison second (see README). For every configuration:

1. Verify the adapter is applied (output or logprob check against a PEFT reference).
2. Warm up, flush caches, fix the request count and seed.
3. Repeat and report variance. A difference smaller than the run-to-run spread is reported as inconclusive.
4. Log the full command line, versions, commit hashes, and GPU state.

## Comparison fairness

Engines are compared as whole systems under the same offered load. Equal flags are not required and often do not exist; what has to be equal is the workload and the tuning effort.

1. Identical offered load: same model, adapters, prompts, arrival process, adapter popularity, and TTFT limit.
2. Fixed tuning protocol: every engine gets the same number of trials from a documented list of settings. The winner is chosen on a separate tuning workload and then measured on a held-out one, so no engine is tuned on its test data and no engine gets more effort than another.
3. Every setting used is recorded, so a reader can see how each engine was run.
4. Relative results: each engine's adapter results are also reported against its own no-adapter baseline. This separates "this engine is slower overall" from "this engine pays more for adapters", and it transfers across engine architectures better than absolute numbers. The definition is in the Decisions table.

The knob-level replication (the `max_loras` claim) only applies to engines that have a matching setting, which excludes mistral.rs.

## Threats to validity

- Engines expose LoRA differently (slots, residency, per-request selection). Equal flags do not mean equal behavior; every mapping between engines is documented and justified.
- mistral.rs has no equivalent of `--max-loras`. Its `--lora-max-adapters` (default 16) caps adapters held in memory, not adapters per batch. Its documentation says requests that use different adapters "run side by side", but not whether they are batched together; that is unknown and is one of the things this study measures. It is excluded from the `max_loras` replication only.
- Effort asymmetry: the author knows some engines better than others. The fixed tuning protocol above is the mitigation, and its trial counts per engine are published.
- Whole-engine effects: SGLang, vLLM, and mistral.rs differ in far more than their LoRA path. Results describe engines as configured, not LoRA kernels in isolation; the relative metric narrows this but does not remove it.
- Versions move quickly. Results describe the pinned versions only.
- One GPU, synthetic traffic, one or two models. Conclusions are limited to that.
- The GPU is a rented, shared machine. Check it is idle before every run and record the check.

## Order of work

1. SGLang and vLLM first: correctness check, replications, then the matched comparison.
2. mistral.rs after that, in the same outcome-level comparison, starting with a check that its adapter path works on the chosen model. If time runs out, the first two publish on their own and mistral.rs is reported as not done.

mistral.rs facts below come from its documentation on master as of 2026-11-21 and have not been confirmed by running it: per-request adapter selection; PEFT-format adapters (`adapter_config.json` plus one `adapter_model.safetensors`); defaults of 16 adapters, rank 256, and 8 GiB of adapter memory; DoRA, aLoRA, QALoRA, bias terms, and adapters that change the embeddings or output head are rejected; adapters cannot be combined with speculative decoding.

## Layout

- `scripts/` load generators, harness, correctness checks
- `configs/` engine and experiment configurations
- `results/raw/` unprocessed run outputs
- `analysis/` code that turns raw results into tables and figures
- `docs/` this plan, the changelog, and notes

## Disclosure

AI coding tools are used throughout. The experimental design, the checks, and the conclusions are the author's. Anything in this repository that the author has not yet reviewed line by line is marked as such.
