# csgmv against triton, one condition at a time from the PR #10286 setup toward ours (plan, fixed before any run)

Added 2026-11-24 about 03:55 UTC, local machine, author approved. Not run. Runs only on the pod in a quiet window declared in the CHANGELOG first.

Why: discussion #52614 got a comment (author not a maintainer) naming four differences between PR #10286 and our runs: one adapter against 32, open loop at saturation against closed loop, 8B against 1.5B, radix cache off against on. These are hypotheses; we varied none of them one at a time.

Design (SGLang 0.5.23, `--lora-backend triton` and `csgmv`, otherwise identical flags, 3 repeats, order of backends alternated per repeat):
- S0, PR #10286 as written: Llama-3.1-8B-Instruct, 1 adapter, `--disable-radix-cache`, `sglang.bench_serving` ShareGPT, 1000 prompts, rates 2, 8, 16, inf. Metric: median E2E latency. If S0 does not move toward the PR's numbers (E2E -9%, -40%, -70%, -81%), the difference is in our setup or the hardware, and later steps are not interpreted.
- S1: S0 on Qwen2.5-1.5B (model size only).
- S2: S1 with 32 Zipf adapters on 16 slots, still open loop, radix off.
- S3: S2 with radix cache on.
- S4: S3 closed loop at concurrency 32 and 128, the study's harness and metrics (the #52614 setup).
Each step changes one condition from the one before. A step "explains" part of the gap only if the change in the csgmv effect between adjacent steps exceeds the spread across repeats (min-max ranges not overlapping). Otherwise it is reported as not distinguished.
Limits: one GPU type (RTX 4090 24 GB; the 8B model needs a fit check first and may need `--mem-fraction-static` changes recorded here), rank and adapter weights as in the study unless S0 uses the PR's adapter, which is a different adapter. Results are descriptive for these settings.
