# Plan

Status: decisions fixed, no measurement run yet. The answers below were fixed on 2026-11-21, before any experiment was run, and are subject to the feasibility checks listed after them. Any change after the first run is recorded in `docs/CHANGELOG.md` with the reason, and this repository's history is not rewritten.

## Decisions

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

## Still open

- Tuning protocol: the list of settings each engine may vary (for example CUDA graphs, chunked prefill, maximum sequences, adapter slots), the number of trials each engine gets, the rule for choosing the winner, and what separates the tuning workload from the held-out one.
- Relative metric: the exact formula for normalizing an engine's adapter results against its own no-adapter baseline.

## Feasibility checks, not yet done

If any check fails, the affected decision is revisited and the change is logged in `docs/CHANGELOG.md` with the reason.

- Each engine can run LoRA on Qwen2.5-1.5B with all of the target modules above (SGLang, vLLM, and mistral.rs on safetensors weights).
- The replication model: the public vLLM sweep used Llama-3.1-8B, which is gated. If it cannot be used, the closest feasible model is used and the replication is labeled as not identical.
- Whether the 500-adapter replication grid fits on 24 GB; if not, the largest feasible grid is used.
- Engine installs on the GPU machine, and enough disk for models and adapters.
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
4. Relative results: each engine's adapter results are also reported against its own no-adapter baseline. This separates "this engine is slower overall" from "this engine pays more for adapters", and it transfers across engine architectures better than absolute numbers.

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
