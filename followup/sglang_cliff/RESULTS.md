# SGLang cliff past the LoRA slot limit: results (Phases 1 and 2)

All runs: Qwen2.5-1.5B-Instruct, 32 adapters (B scale 0.05, checksums equal the manifest), Zipf popularity, 256 in / 128 out, `ignore_eos`, one RTX 4090 (`runpod2`, host 48bd787e389e), 550 requests per run (50 warm-up + 500). Screening runs, FLAGGED. Raw data and tables: `results/phase1/`, `results/phase2/` (`analysis.txt`). Phase 2 is one repeat per cell, so it shows which hypothesis the data supports, not spread.

## Phase 1: is PR #46500 the fix? No.
At 40 rps, p95 TTFT (ms), median of 3 starts, v0.5.23 against main `0e9cff98f5b7`: drainer off 13,379 against 12,029; drain 1.0 11,067 against 8,729; drain 0.2 9,910 against 9,185. No run on either version passed 500 ms (best single run 8,418 ms). By the rule fixed in advance: no material effect for all three settings. At the default (0.0) the drainer is not created, so main equals v0.5.23 there. Full table: `results/phase1/analysis.txt`.

## Phase 2: what the data supports (predictions were fixed in PLAN.md before the runs)
Cells: A = 16 slots 40 rps, B = 32 slots 40 rps, C = 16 slots 20 rps. Tracing overhead (main, 16 slots, 20 rps, 3 patched against 3 unpatched starts): p50 TTFT 35.4 against 36.8 ms, p95 84.7 against 82.7 ms, 2,590 against 2,589 tok/s; inside the 10% limit.

| observable | v0.5.23 A | v0.5.23 B | main A | main B |
|---|---|---|---|---|
| adapter loads / evictions in the window | 27 / 27 | 7 / 6 | 29 / 29 | 8 / 7 |
| loads per second | 1.16 | 0.40 | 0.92 | 0.47 |
| load time per load (CPU time in the call) | 21 ms | 20 ms | 47 ms | 19 ms |
| load time as share of busy window | 2% | 1% | 4% | 1% |
| median distinct adapters per decode batch | 16 | 31 | 16 | 26 |
| mean decode batch size | 143 | 197 | 173 | 127 |
| requests whose adapter was not resident on arrival | 154 of 550 | 6 | 170 of 550 | 7 |
| median wait to first prefill, not resident / resident | 10.0 s / 0.00 s | 0.00 / 0.00 | 13.6 s / 0.00 s | 0.00 / 0.00 |
| mean / p95 wait, all requests | 2.75 / 14.4 s | 0.00 / 0.00 | 4.00 / 18.5 s | 0.00 / 0.00 |
| share of requests waiting over 1 s | 26% | 0% | 27% | 0% |
| output tokens per second | 2,934 | 4,006 | 2,121 | 4,080 |
| Spearman correlation, wait with queue length at arrival | 0.42 | 0.38 | 0.18 | 0.43 |

Predictions against the data (the script's checks are in `analysis.txt`; the C cell for main is not clean, see below):
- **H1, thrashing: contradicted.** Loads are about 1 per second at 16 slots, not 3 or more; 27 to 29 loads in the whole run. The prediction needed at least 3 per second and 10 times the 32-slot rate; neither held, in either version. (32 slots also evicts a handful of times, because the base-model slot counts as a resident entry; the prediction of zero evictions was wrong.)
- **H4, synchronous loads stall the scheduler: contradicted.** Load time is 2% (v0.5.23) and 4% (main) of the busy window at 16 slots, under the 5% line fixed in advance, and about 20 ms per load. (Limit: this is CPU time in the call; an asynchronous GPU copy is not synchronized, so it is a lower bound on the stall.)
- **H3, starvation by adapter: supported in v0.5.23, same pattern in main.** Requests whose adapter was not resident on arrival wait a median 10 to 14 s; resident ones wait 0.00 s. 28 to 31% of requests are in the first group. At 32 slots, almost nobody is (6 to 7 requests) and nobody waits. The wait is only weakly tied to queue position (correlation 0.2 to 0.4, under the 0.5 line). The main A cell was judged by the script as failing H3 only because main's C control cell is not clean (below); the A numbers match v0.5.23's.
- **H2, fragmentation: partly supported.** The decode batch holds 16 distinct adapters (the cap) at 16 slots. Mean decode batch size is 0.73 times that at 32 slots in v0.5.23 but 1.4 times in main, because batch size follows the backlog; this prediction was a weak observable and is "not distinguished" for main.
- **H5, capacity: supported in v0.5.23 (2,934 against 4,006 tok/s, 0.73), and in main (2,121 against 4,080, 0.52).** The 16-slot server completes less work in the same window than the 32-slot server.
- **vLLM comparison: held.** vLLM 0.34.0, 16 slots, 40 rps: mean queue time 0.241 s (p95 bucket at most 1.5 s), 4,676 tok/s, p95 TTFT 1,446 ms. SGLang's mean wait is 2.75 s (v0.5.23) and 4.00 s (main): 11 and 17 times higher; the prediction was at least 5 times. At 20 rps vLLM's mean queue time is 0.009 s. vLLM also loses some TTFT at 40 rps (1.4 s, over the 500 ms bar) but nowhere near SGLang's 14 s.

**What the data supports.** At 16 slots, about 3 in 10 requests need an adapter that is not in the pool; they wait about 10 to 14 s, and the resident adapters' requests do not wait at all. The loads themselves are rare and cheap, so the time is spent waiting for a slot, not loading. Throughput in the same window is 27% to 48% lower than at 32 slots. Admission lets a non-resident adapter in only when the running batch has fewer than 16 distinct adapters (`SOURCE.md` b, `scheduler.py` `can_schedule_lora_req` / `validate_lora_batch`); the traces show 16 distinct adapters in the running decode batch throughout. vLLM has the same skip rule (`SOURCE.md` c) but queues about 11 to 17 times less in this cell. I have not tested why the identical rule gives different waits.

## Unplanned findings
- **main at 16 slots hit the cliff at 20 rps in one traced run** (p95 TTFT 23.2 s, 124 of 500 requests over 1 s, first slow request at 3.6 s; 58 loads, 42 evictions; 1,615 tok/s against 2,560 offered). v0.5.23 at 16 slots, 20 rps in the same session: 83 ms, 0 slow. The same server configuration passed 20 rps in all 6 Phase 1 and Part A starts (default flags, untraced and traced), and drain 0.2 failed once on each version at 20 rps in Phase 1. So 20 rps is not a clean below-cliff control for main: 4 of about 30 20 rps starts on 16 slots tipped into the slow state, 3 of them on main. This is how the C check for main came out invalid. Each of these runs happens once; I do not have the rate of tipping.
- **My Phase 1 explanation of the two 20 rps failures was probably wrong.** I attributed them to pod CPU spikes (external CPU 19 to 22 cores at those starts). The Phase 2 failure started with external CPU at 1.6 cores, on a quiet pod. The spikes may be unrelated. I have not shown what tips the server over at 20 rps.
- Batches in the traces show running decode batches of up to about 170 requests, so the limit that binds is distinct adapters, not batch size.

## Not established
- Why admission with the same rule leaves vLLM's queue short; whether the SGLang waits come from the loop that picks which waiting request to admit (a skipped request keeps its place, with nothing to raise its priority) or from decode steps keeping all 16 adapters present. Phase 3 would test an intervention.
- Whether the 20 rps tipping is a property of the server (a slow state it cannot leave) or of the load; needs repeats at 20 rps.
- Everything here is one GPU, one 1.5B model, a 550-request burst, one repeat per Phase 2 cell. The SGLang v0.5.23 and main environments differ in kernel packages.
