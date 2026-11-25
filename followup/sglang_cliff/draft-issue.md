# DRAFT (not posted): latency cliff when active LoRA adapters exceed `--max-loras-per-batch`

Status: local draft only. Everything below is FLAGGED screening data from a shared pod (one repeat per cell, one GPU). Not for posting without Brian's approval.

## Summary
With 32 adapters (Zipf popularity) at 40 requests/s, p95 TTFT is about 13 s at `--max-loras-per-batch 16` and about 100 ms at 32. Our traces point at decode CUDA graph coverage, not adapter loading or slot ordering: with the default launch on a 24 GB GPU at `--mem-fraction-static 0.85`, decode graphs cover batches of at most 48, and above that each decode step takes about 4x longer, so slots turn over slowly and non-running adapters starve. With `--cuda-graph-max-bs-decode 256` the same 16-slot server gave p95 TTFT 2.1-3.1 s at 40 rps (3 starts) and throughput equal to vLLM's.

## Setup
Qwen2.5-1.5B-Instruct, RTX 4090, rank-16 adapters on q/k/v/o/gate/up/down, 32 adapters, Zipf, open-loop Poisson, 500 requests plus 50 warm-up, seed 1. SGLang v0.5.23 and main `0e9cff98f5b7`.

## What we measured
- Main at 16 slots, 40 rps: p95 TTFT 13.5 s, 23.6 requests/s done. At 32 slots: 94 ms, 31.9.
- Slot sweep on main at 40 rps: 20 slots 11.7 s, 24 slots 9.0 s, 28 slots 107 ms (p99 9.5 s), 32 slots 94 ms.
- At 16 slots about 3 in 10 requests need an adapter that is not running; they wait a median 10 to 14 s. Requests for running adapters wait about 0 s.
- Loads are about 1 per second at about 20 ms each, 2 to 4% of busy time. Slot thrashing and load stalls are not the cause.
- Calls to the LoRA admission check: 99.9% found no free slot.

## Fix tested (flag only)
- main, 16 slots, 40 rps, `--cuda-graph-max-bs-decode 256`: p95 TTFT 2.08, 3.14, 2.18 s (stock 13.4-13.5 s); TPOT p50 12-14 ms (stock 43 ms); requests/s 36.7, 35.7, 36.5 (stock 23.6). vLLM 0.34.0 same load: 1.45 s, 10 ms, 36.7.
- The decode call CPU time jumps from 1.9 ms to 31.7 ms at batch size 49 in stock, exactly one above the largest captured graph.

## Second effect once decode is fast
- With the graph limit raised, each adapter load is followed by a decode interval 51-58 ms longer than the 6.4 ms median; about 24% of the busy window at 40 rps and 16 slots (CPU timestamps, 3 starts, shared host). Not yet separated into host-side copy and GPU wait.

## What did not help (all main `0e9cff98f5b7`, 16 slots, 40 rps)
- PR #46500's drainer (`--lora-drain-wait-threshold` 1.0 and 0.2): p95 8.8 s and 8.5 s. It moves waiting from non-running to running adapters; mean wait about the same.
- Depth-first admission patch (a free slot goes to the adapter with the most waiting requests, 10 s age cap): p95 13.50 s against 13.46 s stock. The rule rejected 11 requests in 550.
- v0.5.23 against main with default flags: no material difference.

## Open
- vLLM 0.34.0 with the same skip rule and 16 slots had mean queue time 0.24 s against 2.75 s (v0.5.23) and 4.00 s (main). We have not found why.
- At 16 slots and 20 rps, 4 of 27 starts so far tipped into a slow state (p95 above 10 s); trigger unknown.

## Question for maintainers
Should the default decode graph range be larger when LoRA is enabled (or sized from free memory after the adapter pool), and should a decode batch above the captured range be logged? A 16-slot LoRA server falls to about 4x slower steps at 49 concurrent requests.
