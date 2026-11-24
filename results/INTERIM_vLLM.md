# Interim result: vLLM 0.34.0 only (study in progress)

Status: partial. This is the first engine to finish. SGLang and mistral.rs have not finished, the two replications have not run, and adapter correctness is not certified for any engine. Nothing here is a comparison between engines. Written at study hour about 5.9 (2026-11-22, about 05:15 UTC); the study's hard stop is 23:25:58 UTC the same day.

## Setup

- One RTX 4090 (24 GB), vLLM 0.34.0, Qwen2.5-1.5B-Instruct, bf16. Tuning settings are the defaults (the three tuning trials were within about 0.1% of each other; see `results/raw/tuning/`).
- LoRA runs use `--max-loras 16`. Adapters are synthetic (rank 16, all linear layers, random weights), so these runs show serving cost, not the quality of any real fine-tune.
- Open-loop Poisson arrivals, 256-token prompts, 128 output tokens, 500 requests per run (the first 50 excluded), 3 repeats per step. "Passing" means the median across repeats of p95 time-to-first-token (TTFT) is 500 ms or less.
- Each session started and ended with a reference run; drift was -0.2% (LoRA off) and +0.7% (LoRA on), inside the 5% rule.
- Raw per-run files are in `results/raw/main/vllm/`.

## Result: highest offered rate that passed, and what happened next

| Workload | Last passing rate (rps) | Next rate: p95 TTFT range across 3 repeats |
|---|---|---|
| LoRA off (b0) | 80 | 160 rps: 892 to 1037 ms |
| LoRA on, no adapter used (b1) | 80 | 160 rps: 962 to 1058 ms |
| 1 adapter | 80 | 160 rps: 1565 to 2024 ms |
| 32 adapters, Zipf popularity | 20 | 40 rps: 958 to 1565 ms |
| 32 adapters, uniform popularity | 10 | 20 rps: 496 to 938 ms (one repeat was under the limit) |
| shared prefix, no adapter | 80 | 160 rps: 780 to 929 ms |
| shared prefix, 32 adapters, uniform | 10 | 20 rps: 492 to 931 ms (one repeat was under the limit) |

The ladder is coarse (5, 10, 20, 40, 80, 160, 320 rps), so each "last passing rate" is only bounded by its neighbors.

## What this does and does not show

- Turning LoRA on without using an adapter made no measurable difference to latency or throughput (b1 against b0).
- One adapter cost a modest amount: throughput at 160 rps was about 20% lower than without LoRA.
- With 32 adapters sharing 16 slots, the rate that stays under the limit dropped from 80 to about 10 to 20 rps. Zipf traffic fared better than uniform.
- Shared-prefix results with 32 adapters match the plain 32-adapter results, so this run shows no extra cost from prefix sharing, and cannot separate any such cost from the adapter-count effect.
- Why it happens is not established. Reading the vLLM source suggests the scheduler allows at most `max_loras` distinct adapters per step, so requests for other adapters wait; adapter activation is also synchronous. Both are unmeasured. The planned `max_loras` sweep is the test: if 32 slots remove the slowdown, this is a configuration effect, not a defect.
- A first complete run of this grid was lost with its raw files when the machine's network volume lost its contents mid-study (cause unknown; see the changelog). Its per-step summaries, kept in `results/raw/main_firstrun_logs/`, agree with this rerun to within a few percent. The rerun is the one reported.

## Limits

One engine, one GPU, one model, synthetic adapters, 3 repeats, a single day, and a coarse rate ladder. No claim is made about other engines, other hardware, or real fine-tuned adapters.
