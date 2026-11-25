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

## Option 4 result: would a reuse predictor pick better drain victims? (offline replay, FLAGGED)
`replay_slots.py` on the 40 rps request lists of six 16-slot starts; plan `8148574` committed before the replay; output `results/phase2/replay_slots.txt`. Decode time per request is fixed at its measured value, so batch-size effects on step time are not modeled.

| median over 5 normal cells | p95 TTFT | mean wait | makespan |
|---|---|---|---|
| measured | 13.3 s | 2.06 s | n/a |
| F (FCFS skip) | 13.06 s | 2.03 s | 22.5 s |
| R1 drain, soonest-to-empty victim | 9.49 s | 3.55 s | 28.3 s |
| R2 victim by expected arrivals (30 s rate) | 9.36 s | 3.51 s | 27.9 s |
| R3 victim by lowest rate | 9.46 s | 3.57 s | 28.1 s |
| R4 oracle (actual future arrivals) | 9.29 s | 3.43 s | 28.0 s |

- V1 holds: the model's FCFS matches the measured p95 and mean wait within 2% in all 5 normal cells.
- V2 holds: R1 lowers p95 by 27% (the measured drainer was 35% lower), and raises mean wait by about 75% and makespan by about 25%. The measured drainer also raised mean wait (3.3 s against 2.03 s stock).
- V3 holds: R2 is 1% and R3 is 0% below R1 on p95. A rate-based reuse predictor does not choose better victims here.
- V4 holds: the oracle, which knows the true future arrivals, is 2% below R1. No predictor can gain much on this workload.
- The slow-state start (U4) behaves differently: draining there makes p95 worse (15.8 s against 14.8 s) and mean wait much worse.
- Reading: the drain trades tail latency for mean latency and total completion time; which running adapter is drained barely matters. This is a model of the scheduler on one workload with fixed decode times.

## Option 5 result: decode CUDA graph limit raised (part H), FLAGGED screening
`runpod2` host 48bd787e389e; plan `af0fb3c` committed before the runs; stock main `0e9cff98f5b7`, 16 slots, only `--cuda-graph-max-bs-decode 256` changed; three starts, seed 1; stock references E U1, U5 (normal state) and U4 (slow state); vLLM from Part C.

| 40 rps | stock main (U1, U5) | graph limit 256 (3 starts) | vLLM 0.34.0 |
|---|---|---|---|
| p95 TTFT | 13.44, 13.46 s | 2.08, 3.14, 2.18 s | 1.45 s |
| p99 TTFT | 15.4 s | 2.5, 3.7, 2.9 s | 1.8 s |
| TPOT p50 | 43.3, 43.7 ms | 12.1, 14.2, 12.4 ms | 10.0 ms |
| requests/s | 23.6 | 36.7, 35.7, 36.5 | 36.7 |
| tok/s | 3011, 3018 | 4694, 4569, 4662 | 4676 |
| decode CPU time, 49+ requests | 31.6 ms | 2.0, 2.5, 2.0 ms | n/a |

20 rps: all three starts passed (p95 95, 101, 81 ms), 0 errors.
Predictions (PLAN.md option 5): H1 (TPOT below 20 ms) holds; H2 (p95 at least 3x lower, at most 4.5 s) holds (6.5x, 4.3x, 6.2x); H3 (requests/s at least 30.7) holds; H4 (decode CPU time under 10 ms at 49+) holds; H5 (20 rps passes 3 of 3) holds. Not at the study's 500 ms bar at 40 rps (2.1 to 3.1 s).
Reading: with the graph range raised, SGLang's 16-slot throughput matches vLLM's (36.7 requests/s, about 4.7k tok/s) and TPOT falls from 43 to 12-14 ms. The step time above 48 requests, not the slot rule, was the main reason a slot stayed held 5 times longer. A gap to vLLM remains in p95 TTFT (2.1-3.1 s against 1.45 s). I have not run the 32-slot cell with the larger range, nor tested whether main without LoRA behaves the same, nor the default on other GPUs.
Consequences: the earlier readings "every scheduling lever is exhausted" and "the cliff is a slots-versus-adapters threshold" were incomplete: the slot count matters, but the capacity lost at 16 slots was largely this step-time penalty. Page claim "same slots, same traffic" was not the same setting; a line was added to "Things I got wrong" in `docs/DRAFT_POST.md` and `docs/DRAFT_POST_long.md`. Single flag, shared host, screening only; any timing claim for the page needs the dedicated machine.

## Option 2 read: what is left of the gap to vLLM (from existing traces, no new runs, FLAGGED)
At 16 slots and 40 rps, graph limit 256 (3 starts) against stock and vLLM:
- SGLang with the limit raised: mean wait 0.35, 0.46, 0.37 s; p95 wait 2.0, 3.0, 2.1 s; non-running adapters' median wait 1.2, 1.5, 1.2 s; mean decode batch 52 to 57 (max 81 to 92); 72 to 84 loads. Stock: mean wait 2.03 s, p95 13.1 s, non-running median 9.5 s, mean batch 102.
- vLLM: mean queue 0.24 s, p95 TTFT 1.45 s, TPOT 10 ms. SGLang with the limit: TPOT 12 to 14 ms.
- Reading: a slot is held about one request lifetime (128 tokens), which is 1.5 to 1.8 s in SGLang here and 1.3 s in vLLM, and the wait of a non-running adapter is about one to two lifetimes. A 20 to 40% longer step plus the spread over three starts is the size of the remaining gap. I cannot separate those from this data. The decode-call CPU time in the trace is 2 ms in both and says nothing about GPU step time.
- Not tested: GPU step time of the LoRA kernels in each engine (needs a profile), and repeats on the dedicated machine.

## Option 6 result: step-time microbench (I) and rerun (J) on `runpod2`, FLAGGED
Host 48bd787e389e; plan `c489bca` committed before the runs; untraced. External CPU at the 15 starts: under 5 cores at 14 of them; `J_C32-r2` had 22.6, 23.4 and 14.5 (its 40 rps p95 was 53 ms, so no visible effect). So these runs are mostly quiet but not a dedicated-machine measurement; any page number still needs the dedicated session.

**I, TPOT p50 in ms at fixed concurrency (300 requests, 0 errors in all):**

| | N=16 base | N=16 16 adapters | N=48 base | N=48 16 adapters | N=96 base | N=96 16 adapters |
|---|---|---|---|---|---|---|
| SGLang stock | 4.9 | 5.8 | 5.7 | 6.8 | 33.5 | 33.5 |
| SGLang graph limit 256 | 4.9 | 5.8 | 5.7 | 6.7 | 7.1 | 8.3 |
| vLLM 0.34.0 | 4.6 | 7.2 | 5.3 | 8.6 | 6.6 | 10.3 |

- M1 holds: stock at N=96 is 4.0 times the raised limit (33.5 against 8.3; needed 2.5), and at N=16 they are equal. The graph limit is the whole step-time effect, and it shows without any adapter swapping (base model, N=96: 33.5 against 7.1).
- M2 fails, and in the other direction: with 16 resident adapters SGLang's step is shorter than vLLM's (ratio 0.81, 0.78, 0.81 at N=16, 48, 96). On the base model SGLang is 1.08 to 1.10 times vLLM. So my statement that the remaining gap looked like a 20 to 40% longer SGLang step was wrong for the all-resident case.
- Open-loop at 40 rps with 32 adapters, TPOT is 12 to 14 ms for SGLang (raised limit) against 10 ms for vLLM, while with 16 resident adapters SGLang is 6.7 to 8.3 ms against vLLM's 8.6 to 10.3. So the extra SGLang step time under the Zipf-32 load appears only when adapters are being swapped. This is consistent with the swap cost (72 to 84 synchronous loads in the traced raised-limit runs) but I have not measured it as the cause.

**J, 40 rps / 20 rps, median over 3 starts (p95 TTFT; requests/s):**

| config | 40 rps p95 (each start) | requests/s | 20 rps p95 (each start) |
|---|---|---|---|
| S16 stock | 13.50 s (13.50, 13.45, 13.63) | 23.5 | 83 ms (91, 79, 83) |
| C16 limit 256 | 2.18 s (2.15, 2.43, 2.18) | 36.4 | 88 ms (93, 88, 82) |
| S32 stock | 98 ms (106, 95, 98) | 31.3 | 48 ms |
| C32 limit 256 | 63 ms (686, 53, 63) | 39.2 | 46 ms |

- J1 holds (each C16 start at most 4.5 s, 6.2 times lower than S16). J3 holds (20 rps passed 3 of 3 for C16, S32, C32). J4 holds (36.4 against 23.5 is 1.55 times).
- J2 fails: one C32 start (round 1) had p95 686 ms (p99 958 ms), above 500 ms. Its external CPU at start was under 5 cores, so I have no explanation; the other two C32 starts were 53 and 63 ms.
- Unplanned: stock at 32 slots also completes fewer requests (31.3 per second against 39.2 with the limit raised) at the same 40 rps offered, though its p95 passes. So the graph range costs throughput at 32 slots too.
- Not tested: whether the swap cost explains the remaining 16-slot gap to vLLM; the dedicated machine; other GPUs and memory settings.

## Option 7 result: swap cost around adapter loads (existing traces, FLAGGED)
`swap_cost.py` on `H_main-16-cg256-1..3` (graph limit 256, 16 slots, 40 rps) and stock `E_main-16-U1`; plan `ce8a3ee` committed before the analysis; output `results/phase2/swap_cost.txt`.

| cell | decode intervals, no load / with load | median interval no load | median interval with a load | load call time | excess time in load intervals |
|---|---|---|---|---|---|
| graph limit 256, start 1 | 1265 / 67 | 6.4 ms | 57.9 ms | 1.64 s (10.8% of 15.1 s) | 3.65 s (24.2%) |
| graph limit 256, start 2 | 1166 / 60 | 6.4 ms | 64.5 ms | 1.57 s (10.1%) | 3.57 s (23.0%) |
| graph limit 256, start 3 | 1261 / 68 | 6.4 ms | 57.2 ms | 1.66 s (10.9%) | 3.58 s (23.6%) |
| stock, start U1 | 663 / 23 | 33.7 ms | 70.1 ms | 0.73 s (3.2%) | 0.97 s (4.3%) |

- S1 holds: the median decode step without a load is 6.4 ms (at most 9), close to the resident microbench. S2 holds: a decode interval that contains a load is 51 to 58 ms longer in the median (needed 15). S3 holds: the excess is 23 to 24% of the busy window (needed 8%).
- Reading: with the graph limit raised, adapter loads stall decoding for about 50 to 60 ms each, of which about 20 ms is the CPU time inside the load call. About a quarter of the busy window is spent in those stalls. Without the stalls the average step would be near the resident microbench. This is the best explanation I have for SGLang's 12 to 14 ms TPOT against vLLM's 10 ms, but it is from CPU timestamps, not GPU-synchronized, and it does not show how vLLM avoids the stall.
- This changes one earlier reading: Phase 2 found loads contradicted as the cause (3% of busy time) in the stock server, where slow steps hide them. With the graph limit raised the same loads take 24% of the window.
- Not tested: what part of the 50 to 60 ms is host-side weight preparation (`pin_memory` per load), what part is waiting on the GPU, and whether `--enable-lora-overlap-loading` removes it.

## Option 8 result: overlap loading on top of the raised graph limit (K2), FLAGGED
`runpod2` host 48bd787e389e; plan `2f91385` plus amendment 3 `c4fcf68` (all three first starts, `K_main-16-cg256-ovl-1..3`, failed at launch on `--max-loaded-loras`; no measurement) committed before the runs; flags `--cuda-graph-max-bs-decode 256 --enable-lora-overlap-loading --max-loaded-loras 32`; traced; external CPU at the three starts 1.0 to 3.0 cores.

| 40 rps | graph limit 256 (H, 3 starts) | plus overlap loading (K2, 3 starts) |
|---|---|---|
| p95 TTFT | 2.08, 3.14, 2.18 s | 1.92, 1.92, 1.97 s |
| p99 TTFT | 2.5, 3.7, 2.9 s | 2.27, 2.32, 2.40 s |
| TPOT p50 | 12.1, 14.2, 12.4 ms | 11.3, 11.4, 11.5 ms |
| requests/s | 36.7, 35.7, 36.5 | 37.1, 36.9, 37.1 |
| decode interval with a load (median) | 57.9, 64.5, 57.2 ms | 50.5, 50.4, 51.6 ms |
| excess time in load intervals | 24.2, 23.0, 23.6% | 21.2, 21.1, 21.5% |

20 rps: p95 116, 115, 116 ms, 0 errors (reference H: 95, 101, 81 ms).
Predictions: K1 (median p95 at most 1.8 s, each start below 2.18 s) half holds: each start is below 2.18 s but the median is 1.92 s, not 1.8 s, so K1 fails as stated. K2 (requests/s at least 36.4 in each start) holds. K3 (20 rps passes 3 of 3) holds. K4 (excess at most 12%) fails: 21.1 to 21.5%.
Reading: overlap loading gives a small gain (p95 about 12 to 15% lower, TPOT about 1 ms lower, throughput about 1% higher) and much tighter run-to-run spread, but it does not remove the stall: a decode step that follows a load is still about 44 ms longer than one that does not, and about a fifth of the busy window is spent in those steps. So the stall is not mainly the host-to-device copy of the weights. vLLM is still ahead (p95 1.45 s, TPOT 10 ms). Not tested: what else runs in the load path (adapter memory pool writes, eviction, graph input setup), a GPU-synchronized timeline.

## Option 9 result: graph range default, and the slow state against batch size (FLAGGED)
Plan `f59a35f`, runner part L and `batch_share.py` committed before the launches and the analysis. `runpod2` host 48bd787e389e.

**9.1 Launch-only check** (stock main `0e9cff98f5b7`, no load; `results/phase2/L*_*/server.log`):

| launch | `mem_fraction_static` used | decode graphs captured | free memory after capture |
|---|---|---|---|
| L1 LoRA as in the study, no memory flag | 0.704 (SGLang's choice) | 1, 2, 4, 8, 12, 16, 24, 32, 40, 48 | 5.36 GB |
| L2 no LoRA, `--mem-fraction-static 0.85` | 0.85 | same, up to 48 | 2.10 GB |
| L3 no LoRA, no memory flag | 0.704 | same, up to 48 | 5.44 GB |

Prediction held: the largest captured decode batch is 48 in all three, and SGLang's own default fraction is below 0.85 (0.704). So 48 is SGLang's default decode graph range for this 24 GB GPU; it is not caused by my 0.85 and not by LoRA, and more free memory (5.4 GB) did not enlarge it. `cuda_graph_max_bs_decode` is `None` in `server_args` (resolved later). Not tested: other GPUs, and whether the default depends on the GPU class or on something else.
**Correction:** I said the range was short "probably because I left about 2 GB free at capture time with 0.85" (in conversation, and as a parenthetical in the "Things I got wrong" lines of both draft posts). That explanation was wrong; the draft-post lines were edited and a CHANGELOG correction entry added.

**9.2 Slow state against batch size** (`batch_share.py`, traced starts, `results/phase2/batch_share.txt`):
- (a) holds: at 20 rps the two slow starts (`B_main-16`, `E_main-16-P1`; p95 23.2 and 16.9 s) spent 57% and 55% of decode calls above 48; the 9 normal stock starts spent 0%.
- (b) mine holds, the author's does not: at 40 rps the two slow starts (`B_main-16` 2121 tok/s, `E_main-16-U4` 2352 tok/s) spent 94% and 75% above 48, but the 9 normal stock starts spent 70 to 74%, so the share does not separate them. What does differ: the slow 40 rps starts' batches peaked at 355 and 283 and had runs of 91 and 63 decode calls above 256; the normal 16-slot stock starts peaked at 213 to 217 and never exceeded 256. Not a clean separator: `B_v23-16` (2934 tok/s) peaked at 307 and `B_main-32` and `G_main-28` also exceeded 256 without a slow state.
- (c) holds: the 12 raised-limit windows (`H_*`, `K2_*`) peaked at a batch of 92 and never had a decode call above 256.
- (d) fails: in both slow 20 rps starts the first request that waited over 1 s arrived 3.6 and 3.7 s after the first arrival (queue length 1 and 7, non-resident adapter), before the decode batch ever stayed above 48 (from about 13 to 15 s into the window). So the batch above 48 follows the stall; it is not what starts it at 20 rps.
- Unplanned: in both slow 20 rps starts the loadgen's first arrival came late (8.5 and 9.7 s after the mark; 5.8 to 6.7 s in all normal starts), and loads stopped after the first seconds (30 to 31 loads in the first 5 s, then 35 to 37 by 20 s, against 119 to 124 in normal starts) while non-resident arrivals kept coming (117 to 121). Two slow starts only. The late first arrival may mean the host was slow at the start; the earlier CPU-spike explanation was withdrawn, and this does not restore it. The cause is not identified.

## Option 10 result: slot-capture test for the slow state (existing traces, FLAGGED)
Plan `8ba8516`, amendment 4 `630cf3d` (first run matched arrivals to loadgen requests by order and failed its own check in 24 of 33 windows; kept as `results/phase2/slot_capture_v1_positional.txt`; the rerun matches per adapter and passes all checks: one-to-one map, per-adapter counts equal, resident set never above 16). Script `slot_capture.py`, output `results/phase2/slot_capture.txt`. "After 5 s" = 5 s after the first arrival.

| group | resident-set changes after 5 s | evictions after 5 s | decode calls with every resident adapter running |
|---|---|---|---|
| 20 rps slow (`B_main-16`, `E_main-16-P1`) | 24, 29 | 28, 31 | 78%, 80% |
| 20 rps normal stock (7 starts) | 121 to 123 | 123 to 128 | 13 to 23% |
| 40 rps slow (`B_main-16`, `E_main-16-U4`) | 12, 13 | 18, 22 | 91%, 89% |
| 40 rps normal stock (7 starts) | 10 to 14 | 17 to 22 | 86% (all seven) |
| drainer starts (`D_*`), 40 rps | 40, 47 | 53, 53 | 78%, 82% |
| raised-limit starts (`H_*`, `K2_*`), 40 rps | 36 to 50 | 40 to 51 | 76 to 88% |

- (a) fails: slow 20 rps starts changed their resident set 24 and 29 times after 5 s (predicted at most 2). The normal part holds (121 to 123, at least 10).
- (b) fails: 78 and 80% at 20 rps (predicted at least 90%); 91 and 89% at 40 rps. The normal part holds at 20 rps (13 to 23%, at most 50%) and fails at 40 rps (86%).
- (c) fails: 28 and 31 evictions at 20 rps, 18 and 22 at 40 rps (predicted at most 5).
- Reading: the strict form (slots frozen, no turnover) is contradicted. The slow 20 rps starts do turn slots over, but at about a fifth of the normal rate (24 to 29 changes against 121 to 123 in the same time), with nearly all slots busy most of the time (78 to 80% against 13 to 23%): turnover is slow, not stopped. At 40 rps there is no separation: slow and normal stock starts both have about 86 to 91% of decode calls with every slot busy and 10 to 14 changes.
- Unplanned, who held the slots: in normal 20 rps starts the hottest adapters held a slot almost all the time (rank 1 and 2 at 100%) and the tail rotated (50 to 60%); in both slow 20 rps starts even ranks 1 and 2 held a slot only 80 to 91% of the time, and low-ranked adapters held slots for long periods (rank 14 at 97% and rank 26 at 78% in `B_main-16`; ranks 17, 18, 21, 22, 24 at 51 to 78% in both). So in the slow state slots are tied up by mid-ranked and tail adapters while hot adapters are repeatedly pushed out. Two starts only; not a mechanism test.
- Drainer starts: slot turnover is higher than stock at 40 rps (40 and 47 changes against 10 to 14), unchanged at 20 rps (121 and 122). That agrees with the drainer moving waiting around without adding capacity.
- Not tested: why the slow 20 rps starts begin with fewer loads in the first seconds.

## Option 11 result: queue-aware eviction check (existing traces, FLAGGED)
Plan `53394a6` committed before the analysis; script `evict_queue.py`, output `results/phase2/evict_queue.txt`. Eviction rule: `mem_pool.py:810-856` (candidates are pool adapters not in the current batch and not pinned, victim chosen by `lora_eviction_policy` = `lru`, `eviction_policy.py:47,64-76`); the queue is not consulted.

| group (after 5 s) | starts | evictions | evicted adapter had at least 1 request waiting |
|---|---|---|---|
| 20 rps slow (`B_main-16`, `E_main-16-P1`) | 2 | 59 | 0 (0%) |
| 20 rps normal stock | 7 | 868 | 2 (0%) |
| 20 rps drainer | 2 | 247 | 0 |
| 20 rps raised limit | 6 | 742 | 1 (0%) |
| 40 rps slow (`B_main-16`, `E_main-16-U4`) | 2 | 40 | 0 |
| 40 rps normal stock | 7 | 139 | 0 |
| 40 rps drainer | 2 | 106 | 73 (69%) |
| 40 rps raised limit | 6 | 289 | 2 (1%) |

- Predictions fail: 0% of evictions in the slow 20 rps starts hit an adapter with a waiting request (predicted at least 30%), and no top-4 adapter was evicted with any waiting request (predicted 5 or more). The normal part holds (0%, at most 10%). By the plan, no patch plan is written.
- Why it comes out this way: a waiting request for an adapter that is already in the pool is admitted whenever the batch has room (`scheduler.py` `can_schedule_lora_req` counts distinct adapters in the batch, not pool residents), so an adapter is only evicted while idle and with nothing queued for it. LRU is not throwing out adapters that requests are waiting for.
- The top-4 adapters in the slow starts were evicted 18 to 36 s after the first arrival, within about 2 s of each other and with 0 waiting. The arrival phase of a run ends about 27 s after the first arrival (550 requests at 20 rps), so these are evictions during the drain at the end of the run, not part of the slow state.
- Unplanned: the drainer at 40 rps evicts an adapter with requests waiting for it in 69% of its evictions (73 of 106; 36 and 37 per start), against 0 in stock. That fits what the drainer does: it blocks admission for the draining adapter, its queued requests wait, then it is evicted when it empties, and those requests need a reload. It also had about three times the loads of stock at 40 rps.
- The queue-aware eviction patch is not planned: the check it rests on failed.

## Not established
- Why admission with the same rule leaves vLLM's queue short; whether the SGLang waits come from the loop that picks which waiting request to admit (a skipped request keeps its place, with nothing to raise its priority) or from decode steps keeping all 16 adapters present. Phase 3 would test an intervention.
- Whether the 20 rps tipping is a property of the server (a slow state it cannot leave) or of the load; needs repeats at 20 rps.
- Everything here is one GPU, one 1.5B model, a 550-request burst, one repeat per Phase 2 cell. The SGLang v0.5.23 and main environments differ in kernel packages.

## Phase 3 option 9 (plan: PLAN.md, option 9; the time is the commit's)

### 9.1 Launch-only check (part L, `runpod2` host 48bd787e389e, stock main `0e9cff98f5b7`, no load; FLAGGED screening, one launch each)
| launch | `mem_fraction_static` chosen | decode CUDA graph capture (`Capture target decode CUDA graph end. bs=[...]`) | `max_total_num_tokens` |
|---|---|---|---|
| L1: LoRA as in the study, no `--mem-fraction-static` | 0.704 | 1, 2, 4, 8, 12, 16, 24, 32, 40, 48 | 474,066 |
| L2: no LoRA, `--mem-fraction-static 0.85` | 0.85 | 1, 2, 4, 8, 12, 16, 24, 32, 40, 48 | 620,315 |
| L3: no LoRA, no `--mem-fraction-static` | 0.704 | 1, 2, 4, 8, 12, 16, 24, 32, 40, 48 | 494,181 |
- The prediction held: the largest captured decode batch is 48 in all three launches. 48 is SGLang's default on this 24 GB GPU (RTX 4090, 4 MiB used before each launch); it is not a result of my 0.85 and not caused by LoRA. The default memory fraction chosen is 0.704, below 0.85.
- Logs: `results/phase2/L1_lora-default/server.log`, `L2_nolora-085/`, `L3_nolora-default/`. This tests the default of this build on this GPU only.

### 9.2 Slow state against batch size (existing traces, no GPU; `batch_share.py`, output `results/phase2/batch_share.txt`; FLAGGED screening starts)
Predictions as written in PLAN.md, and what the data show:
- (a) At 20 rps the two slow starts (`B_main-16`, `E_main-16-P1`) spend 57% and 55% of decode calls above batch 48; the nine normal stock-limit starts spend 0%. Held.
- (b) At 40 rps the slow starts spend 94% and 75% above 48, and the normal stock starts 57% to 74% (median 71%). The share does not separate slow from normal at 40 rps, as the plan predicted; the author's expectation that it would did not hold.
- (c) The twelve raised-limit windows (`H_*`, `K2_*`): the longest run of decode calls above 256 is 0 and the largest batch is 92. Held.
- (d) The first request over 1 s TTFT comes after the batch has been above 48 for at least 5 s: NOT held. `batch_share.py` compares two clocks (the load generator's `t_send` and the timeline from `marks.jsonl`, which starts before the load generator does); `batch_share_align.py` puts both on the load generator's clock. At 20 rps the first request over 1 s TTFT was sent at 3.6 s (`B_main-16`) and 3.4 s (`E_main-16-P1`) while the mean batch was 29 and 24, and the mean batch first exceeded 48 at 5 s. So slow requests appear before the batch passes 48, not after 5 s above it. (The first output of `batch_share.py`, 3.6 against 13 s and 3.4 against 15 s, mixes the two clocks and should not be read.)
- Reading: slow starts do spend most decode calls above 48 where normal starts do not (20 rps), which fits the graph limit as the cause of slow steps once the batch is large. The traces do not show that passing 48 triggers the slow state: slow requests begin earlier, during the ramp, and at 40 rps the share above 48 is high in normal starts too. The trigger is not established. Limits: traced starts only, one start per slow case, 2 slow starts per rate, a decode call's `n` is the batch at the time of the call.

## Slow-state investigation: stopped, open (decision by the author in the conversation; the time is the commit's)
The investigation of what tips a start into the slow state stops here. It is recorded as open, not solved.
- Ruled out as the trigger: a decode batch above 48 (slow requests begin before the batch passes 48 at 20 rps; at 40 rps the share above 48 is high in normal starts too, 70 to 74% against 94% and 75%); slot capture in its strict form (option 10); the LRU eviction order evicting adapters that have waiting requests (0 of 59 evictions after the first 5 s in the slow 20 rps starts, option 11).
- The raised limit (`--cuda-graph-max-bs-decode 256`) avoided the slow state in every start so far. Four of 27 stock 20 rps starts tipped.
- Not found: the trigger. No further runs are planned for it.
