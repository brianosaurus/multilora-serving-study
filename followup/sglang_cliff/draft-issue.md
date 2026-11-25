# DRAFT (not posted): latency cliff when active LoRA adapters exceed `--max-loras-per-batch`

Status: local draft only. Everything below is FLAGGED screening data from a shared pod (one repeat per cell, one GPU). Not for posting without Brian's approval.

## Summary
With 32 adapters (Zipf popularity) at 40 requests/s, p95 TTFT is about 13 s at `--max-loras-per-batch 16` and about 100 ms at 32. The cause in our traces is not adapter loading; it is that all slots are full almost all the time, so requests for an adapter that is not running wait for a slot, and total throughput is lower.

## Setup
Qwen2.5-1.5B-Instruct, RTX 4090, rank-16 adapters on q/k/v/o/gate/up/down, 32 adapters, Zipf, open-loop Poisson, 500 requests plus 50 warm-up, seed 1. SGLang v0.5.23 and main `0e9cff98f5b7`.

## What we measured
- Main at 16 slots, 40 rps: p95 TTFT 13.5 s, 23.6 requests/s done. At 32 slots: 94 ms, 31.9.
- Slot sweep on main at 40 rps: 20 slots 11.7 s, 24 slots 9.0 s, 28 slots 107 ms (p99 9.5 s), 32 slots 94 ms.
- At 16 slots about 3 in 10 requests need an adapter that is not running; they wait a median 10 to 14 s. Requests for running adapters wait about 0 s.
- Loads are about 1 per second at about 20 ms each, 2 to 4% of busy time. Slot thrashing and load stalls are not the cause.
- Calls to the LoRA admission check: 99.9% found no free slot.

## What did not help (all main `0e9cff98f5b7`, 16 slots, 40 rps)
- PR #46500's drainer (`--lora-drain-wait-threshold` 1.0 and 0.2): p95 8.8 s and 8.5 s. It moves waiting from non-running to running adapters; mean wait about the same.
- Depth-first admission patch (a free slot goes to the adapter with the most waiting requests, 10 s age cap): p95 13.50 s against 13.46 s stock. The rule rejected 11 requests in 550.
- v0.5.23 against main with default flags: no material difference.

## Open
- vLLM 0.34.0 with the same skip rule and 16 slots had mean queue time 0.24 s against 2.75 s (v0.5.23) and 4.00 s (main). We have not found why.
- At 16 slots and 20 rps, 4 of 27 starts so far tipped into a slow state (p95 above 10 s); trigger unknown.

## Question for maintainers
Is a limit of distinct adapters per batch below the active working set expected to behave this way, and is there a scheduling change intended to bound waiting for non-running adapters other than the drainer?
