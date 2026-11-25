# DRAFT (not posted): LoRA server falls out of decode CUDA graphs at batch 49 and its latency collapses at 16 slots

Status: local draft only. All numbers are FLAGGED screening runs on one RTX 4090 pod (3 starts per cell unless stated); not a dedicated-machine measurement. Not for posting without Brian's approval.

## Summary
With 32 adapters (Zipf popularity) at 40 requests/s and `--max-loras-per-batch 16`, p95 TTFT is about 13.5 s. At 32 slots it is about 100 ms. The server log shows decode CUDA graphs captured only up to batch size 48. Above 48 concurrent requests each decode step is about 4 to 5 times slower, requests hold their adapter slot about 5 times longer, and requests for an adapter that is not running wait 10 s or more.
Launching with `--cuda-graph-max-bs-decode 256` (no code change) gives p95 TTFT 2.15 to 2.43 s at 40 rps and 36.4 requests/s (stock 23.5), which equals vLLM's throughput on the same load.

## Setup
Qwen2.5-1.5B-Instruct, RTX 4090 (24 GB), `--mem-fraction-static 0.85`, `--context-length 4096`, `--enable-lora --max-lora-rank 16`, all seven projection modules, 32 rank-16 adapters, Zipf popularity, open-loop Poisson arrivals, 500 requests plus 50 warm-up, 256 input and 128 output tokens, seed 1. SGLang main `0e9cff98f5b7` (v0.5.23 behaves the same at 16 slots). vLLM 0.34.0 with `--max-loras 16`.
With this memory fraction the capture log reads `Capture target decode CUDA graph end. bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48], ... avail mem=1.98 GB`. I did not test other memory fractions or GPUs.

## Evidence
- **Step time against batch size** (trace of the decode call, CPU time): about 1.9 ms for 48 or fewer requests, 31.7 ms for 49 or more, flat to 300. Closed loop at fixed concurrency on the base model: 5.7 ms per token at 48 against 33.5 ms at 96 in stock, and 7.1 ms at 96 with the larger range (vLLM 6.6 ms).
- **Open loop, 16 slots, 40 rps, p95 TTFT** (three rounds, interleaved): stock 13.50, 13.45, 13.63 s at 23.5 requests/s; `--cuda-graph-max-bs-decode 256` 2.15, 2.43, 2.18 s at 36.4. At 20 rps all starts pass (p95 about 90 ms).
- **32 slots**: stock 106, 95, 98 ms at 31.3 requests/s; larger range 686, 53, 63 ms at 39.2. Stock 32 slots also completes fewer requests than offered; the larger range does not.
- **With 16 resident adapters** (no swapping) SGLang's step is shorter than vLLM's: TPOT 5.8 / 6.7 / 8.3 ms at concurrency 16 / 48 / 96, against vLLM 7.2 / 8.6 / 10.3.
- Slot sweep on stock main at 40 rps: 16 slots 13.5 s, 20 slots 11.7 s, 24 slots 9.0 s, 28 slots 107 ms (p99 9.5 s), 32 slots 94 ms.

## What is still slower than vLLM
With the larger range at 16 slots, p95 TTFT is about 2.2 s against vLLM's 1.45 s (TPOT 12 to 14 ms against 10 ms).
- Each adapter load is followed by a decode step about 50 to 60 ms longer than the 6.4 ms median; those steps take about 23 to 24% of the busy window (CPU timestamps from a log-only trace, 3 starts).
- Adding `--enable-lora-overlap-loading --max-loaded-loras 32` lowers p95 to 1.92, 1.92, 1.97 s and TPOT to 11.4 ms, but the stall after a load remains (about 50 ms, 21% of the window). So the weight copy is not the main part of it; I have not found what is.

## What did not help (stock main, 16 slots, 40 rps)
- The drainer from PR #46500 (`--lora-drain-wait-threshold` 1.0 and 0.2): p95 8.8 s and 8.5 s. It moves waiting from non-running adapters to running ones; mean wait does not fall.
- A patch that gives a free slot to the adapter with the most waiting requests (10 s age cap): p95 13.50 s against 13.46 s stock. The rule rejected 11 requests in 550, because 99.9% of admission checks found no free slot.
- An offline replay of drain-victim policies (including one that knows the future arrivals) changes p95 by at most 2%.

## Open
- Why vLLM, which skips a request whose adapter does not fit in the same way, has mean queue time 0.24 s while stock SGLang has 2 to 4 s. The graph limit explains most of that, but the gap that remains is not explained.
- At 16 slots and 20 rps, 4 of 27 stock starts tipped into a slow state (p95 above 10 s), and 2 of 4 stock starts at 40 rps ran at 2.1 to 2.4k tokens/s instead of 3k. I did not find the trigger.
- One start at 32 slots with the larger range had p95 686 ms; the other two were 53 and 63 ms.

## Question for maintainers
Should the default decode graph range be larger when LoRA is enabled, or be sized from free memory after the adapter pool is allocated? Should a decode batch above the captured range be logged at warning level? A 16-slot LoRA server loses about 4x per-token speed at 49 concurrent requests, with no message.
