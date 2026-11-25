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
- **main at 16 slots hit the cliff at 20 rps in one traced run** (p95 TTFT 23.2 s, 124 of 500 requests over 1 s, first slow request at 3.6 s; 58 loads, 42 evictions; 1,615 tok/s against 2,560 offered). v0.5.23 at 16 slots, 20 rps in the same session: 83 ms, 0 slow. The same server configuration passed 20 rps in all 6 Phase 1 and Part A starts (default flags, untraced and traced), and drain 0.2 failed once on each version at 20 rps in Phase 1. So 20 rps is not a clean below-cliff control: of 26 server starts at 16 slots that ran 20 rps (18 in Phase 1, 6 in Part A, 2 in Part B), 3 tipped into the slow state (v0.5.23 drain 0.2, main drain 0.2, main default), 2 of them on main, 1 of them with the pod quiet. This is how the C check for main came out invalid. Each of these runs happens once; I do not have the rate of tipping.
- **My Phase 1 explanation of the two 20 rps failures was probably wrong.** I attributed them to pod CPU spikes (external CPU 19 to 22 cores at those starts). The Phase 2 failure started with external CPU at 1.6 cores, on a quiet pod. The spikes may be unrelated. I have not shown what tips the server over at 20 rps.
- Batches in the traces show running decode batches of up to about 170 requests, so the limit that binds is distinct adapters, not batch size.

## Phase 3 diagnostic: main, 16 slots, drainer on (no behavior change; traced)
FLAGGED screening, one repeat per cell, `runpod2` host 48bd787e389e, seed 1. Data in `results/phase2/D_main-16-d1` (threshold 1.0) and `D_main-16-d02` (0.2).

| main, 16 slots | 20 rps p95 TTFT | 40 rps p95 TTFT | 40 rps tok/s | 40 rps loads / evicts | 40 rps mean wait | non-resident median wait | resident median wait |
|---|---|---|---|---|---|---|---|
| drainer off (B) | 23.2 s (slow state) | 18.5 s | 2121 | 29 / 29 | 4.00 s | 13.6 s | 0.00 s |
| drainer 1.0 s | 90 ms | 8.8 s | 3001 | 71 / 71 | 3.29 s | 2.9 s | 3.35 s |
| drainer 0.2 s | 81 ms | 8.5 s | 2964 | 76 / 76 | 3.37 s | 2.6 s | 4.37 s |
| main, 32 slots (B) | 47 ms | 94 ms | 4080 | 8 / 7 | 0.00 s | n/a | 0.00 s |

- The drainer moves waiting from non-resident adapters (13.6 s median) to everyone (resident median up from 0.00 s to 3.4-4.4 s). Mean wait barely changes (4.00 s to 3.3 s). It does not remove the cliff at 40 rps.
- Requests completed per second at 40 rps offered (550-request burst, so a lower bound on capacity): 16 slots 23.2-23.5 with the drainer, 16.6 without, 31.9 at 32 slots. Every 16-slot cell completes work more slowly than it arrives. A change to which request is admitted next cannot make the queue short in that regime unless it also raises throughput.
- Decode batch mean at 40 rps: 87 (drainer) vs 173 (off) vs 127 (32 slots). The drainer lowers concurrency while it drains.
- Single 20 rps cell: both drainer runs passed (81-90 ms). The drainer-off 16-slot 20 rps run in B was a slow-state run. One repeat each, so this is not evidence that the drainer prevents the slow state.
- I have not counted drain starts and ends; the tracer does not log drainer decisions.

## Phase 3 result: depth-first admission (part E), FLAGGED screening
`runpod2` host 48bd787e389e; main `0e9cff98f5b7`; 16 slots; plan `0ecb5ab` committed before the runs; order U P P U U P (starts U1, P2, P3, U4, U5, P6); seed 1; patched server loaded from `/root/src/sglang-lora-depth` (logged at each patched start).

| 40 rps | stock (3 starts) | patched (3 starts) |
|---|---|---|
| p95 TTFT | 13.46 s median (13.44, 13.46, 15.21) | 13.50 s median (13.46, 13.50, 13.51) |
| p99 TTFT | 15.40 s | 15.45 s |
| requests/s completed | 23.6 (23.6, 23.6, 18.4) | 23.6 |
| mean decode batch | 101.7 (101.7, 101.4, 123.9) | 102.0-104.3 |
| loads | 36-37, 35 | 33-38 |
| mean wait | 2.03 s (2.03, 2.03, 2.77) | 2.03-2.04 s |

20 rps: patched p95 81-85 ms, 3 of 3 passed. Stock p95 82, 82 and 214 ms, 0 errors.

Predictions (PLAN.md Phase 3):
- P1 (patched p95 at most 0.5x stock): fails. Ratio 1.00.
- P2 (batch size and requests/s at least 1.15x): fails. Ratio 1.00.
- P3 (p99 not above stock): holds (equal).
- P4 (20 rps all pass): holds.
Per the plan's reading rule, patched within 20% of stock means the explanation "the slot goes to the wrong waiting adapter, so capacity is lost" is contradicted for this rule. The patch is not a fix candidate.

What I checked: the flag and the patched path reach the server environment (expansion of the runner's env string on the pod), and a direct call of the new function returns the intended answers (one free slot: deepest adapter yes, shallower no, aged request yes; two free slots: top two yes). What I did not check: how many times the rule rejected a request in the real runs; the tracer has no counter. The identical numbers fit a rule that rarely changed the choice. A likely reason, not tested: under FCFS the oldest waiting request is usually from an adapter that already has the most waiting requests, so the depth rule picks the same adapter.

Other observations:
- The stock 40 rps cells vary: one start (U4) ran at 18.4 requests/s and 2352 tok/s (p95 15.2 s), the same slow state as the earlier `B_main-16` cell (16.6 requests/s, 2121 tok/s). Two of the four stock main 16-slot 40 rps starts so far were in it. The 20 rps stock U4 also had p95 214 ms. One start in the slow state was not tied to anything I measured (external CPU 0.5-2 at U4 start).
- Throughput at 16 slots is about 23.6 requests/s in the normal state, 31.9 at 32 slots.

## Phase 3 follow-up: does the depth rule ever fire? (Amendment 2, one patched start, FLAGGED)
`E_main-16-P1`, `runpod2` host 48bd787e389e, plan `a47f244` committed before the run. External CPU at start was 27 and 47 cores, so this start is noisy.
- Counter at the end of the run: about 78,000 calls to the rule, 77,894 of them with no free adapter slot (99.9%). Rejections: 11. Aged allowances: 25. So about 100 calls reached the depth test and 11 were rejected (about 10%).
- 40 rps: p95 TTFT 13.56 s, 23.4 requests/s, 0 errors, the same as stock (13.46 s, 23.6). Prediction (at least 5% rejected among tested calls; within 20% of stock) held.
- 20 rps: p95 16.9 s, 15.7 requests/s, 2004 tok/s, 0 errors. This start tipped into the slow state at 20 rps (the prediction for it was not part of the amendment). One more slow start; the cause is still unknown.
- Reading: in 16-slot overload all 16 slots are in use almost all the time, so the scheduler seldom has a free slot to give out and there is almost nothing to reorder. The no-effect result for depth-first admission is real, not a rule that never ran: it ran about 100 times in 550 requests.
- From the traces (CPU time in the decode call, a lower bound): decode calls with 50 to 300 requests all take about 32 ms at both 16 and 32 slots, so batch size, not step time, tracks throughput.

## Phase 3 option 3 result: slot sweep on stock main (FLAGGED screening, one start per cell)
`runpod2` host 48bd787e389e, plan `68101da` committed before the runs, seed 1, traced. 16-slot reference: E stock starts (median 13.46 s p95, 23.6 requests/s); 32-slot reference: `B_main-32`.

| slots | 40 rps p95 | 40 rps p99 | requests/s | tok/s | 20 rps p95 |
|---|---|---|---|---|---|
| 16 | 13.46 s | 15.40 s | 23.6 | 3011 | 82 ms |
| 20 | 11.68 s | 14.18 s | 25.0 | 3191 | 71 ms |
| 24 | 9.00 s | 12.52 s | 28.8 | 3674 | 60 ms |
| 28 | 107 ms | 9.54 s | 30.0 | 3833 | 53 ms |
| 32 | 94 ms | 109 ms | 31.9 | 4080 | 47 ms |

- S1 holds on the p95 rule: 28 slots passes (107 ms, 0 errors); 20 and 24 fail. But 28 slots has p99 9.5 s: more than 1% of requests still waited seconds, so the cliff is just under the edge there, not gone.
- S2 holds: requests/s rises 23.6, 25.0, 28.8, 30.0 and the 28-slot value is 0.94 of the 32-slot value (needed 0.9).
- Reading: at this load the cliff location follows the number of adapters in use at once (about 31 distinct per decode batch at 32 slots). 16 to 24 slots fail; 28 passes p95 with a tail; 32 passes cleanly. One repeat per cell, one GPU, one model, one 550-request burst.

## Not established
- Why admission with the same rule leaves vLLM's queue short; whether the SGLang waits come from the loop that picks which waiting request to admit (a skipped request keeps its place, with nothing to raise its priority) or from decode steps keeping all 16 adapters present. Phase 3 would test an intervention.
- Whether the 20 rps tipping is a property of the server (a slow state it cannot leave) or of the load; needs repeats at 20 rps.
- Everything here is one GPU, one 1.5B model, a 550-request burst, one repeat per Phase 2 cell. The SGLang v0.5.23 and main environments differ in kernel packages.
