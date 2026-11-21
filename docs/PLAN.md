# Plan

Status: draft, written before any measurement. Once the first run starts, changes to this file are recorded in `docs/CHANGELOG.md` with the reason. Nothing is pre-registered until the open decisions below are filled in and committed.

## Decisions still open

These need to be decided and written down before any run:

- Base model: a small, ungated model (1B to 3B class) so many adapters fit on 24 GB.
- Adapter source: real adapters or synthetic ones of the same shape. Rank, target modules, dtype.
- Adapter counts to sweep, and how many stay resident versus swapped.
- Traffic model: arrival process, uniform versus skewed adapter popularity, prompt and output lengths.
- TTFT limit used to define peak throughput.
- Metrics: TTFT, TPOT, end-to-end latency percentiles, throughput, GPU memory.
- Repeats per configuration and how run-to-run spread is reported.
- What counts as "the claim held" for each replication, as numeric thresholds, fixed before running.
- Engine versions to pin, and whether to re-run on a newer release if one ships mid-study.
- Tuning protocol: the list of settings each engine may vary (for example CUDA graphs, chunked prefill, maximum sequences, adapter slots), the number of trials each engine gets, and the rule for choosing the winner.
- Relative metric: how an engine's adapter results are normalized against its own no-adapter baseline.

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
