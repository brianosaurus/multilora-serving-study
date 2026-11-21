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

## Method

Replication first, matched comparison second (see README). For every configuration:

1. Verify the adapter is applied (output or logprob check against a PEFT reference).
2. Warm up, flush caches, fix the request count and seed.
3. Repeat and report variance. A difference smaller than the run-to-run spread is reported as inconclusive.
4. Log the full command line, versions, commit hashes, and GPU state.

## Threats to validity

- Engines expose LoRA differently (slots, residency, per-request selection). Equal flags do not mean equal behavior; every mapping between engines is documented and justified.
- mistral.rs has no equivalent of `--max-loras`; if it is included, the comparison is labeled as not like-for-like.
- Versions move quickly. Results describe the pinned versions only.
- One GPU, synthetic traffic, one or two models. Conclusions are limited to that.
- The GPU is a rented, shared machine. Check it is idle before every run and record the check.

## Layout

- `scripts/` load generators, harness, correctness checks
- `configs/` engine and experiment configurations
- `results/raw/` unprocessed run outputs
- `analysis/` code that turns raw results into tables and figures
- `docs/` this plan, the changelog, and notes

## Disclosure

AI coding tools are used throughout. The experimental design, the checks, and the conclusions are the author's. Anything in this repository that the author has not yet reviewed line by line is marked as such.
