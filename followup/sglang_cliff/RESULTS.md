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

## Trigger step 1 result: client-path check for the slow state (existing traces, FLAGGED)
Plan: PLAN.md, trigger step 1 and Amendment 5. Script `harness_check.py`; output `results/phase2/harness_check.txt`; the invalid first run is kept as `harness_check_v1_absolute_INVALID.txt` (it compared an epoch time with a relative one; its verdicts are void).
What was measured: per 20 rps window (25 traced starts, all matched, none excluded), drift = (server arrival minus first server arrival) minus (`t_send` minus first `t_send`). The absolute client-to-server delay is not recoverable from the existing records, because `t_send` is relative to the load generator's own zero.
- Normal windows (23): median drift 1.65 to 3.57 ms, p99 of |drift| 24 to 80 ms.
- Slow windows: `B_main-16` median 16.0 ms, p99 246 ms, max 393 ms; `E_main-16-P1` median 17.4 ms, p99 367 ms, max 442 ms. Both are outside the normal range.
- First request waiting more than 1 s: `B_main-16` drift 0.7 ms (3.6 s after the first arrival, wait 1.6 s); `E_main-16-P1` drift 155.7 ms (3.7 s, wait 2.7 s).
Predictions: (a') fails (slow windows' median and p99 drift are outside the normal range); (c') fails (155.7 ms in `E_main-16-P1`, above 100 ms).
Reading: in the slow windows, server arrival times fall behind send times by tens to hundreds of milliseconds as the run goes, while the load generator's own send lag stays at 3 to 4 ms (`max_send_lag_s`). The drift alone cannot say where the delay sits: the trace's `arrive` is logged when the scheduler enqueues a request, so a busy scheduler or a slow hand-off from the HTTP front end would show as this drift, and so would a client or network delay. The client send lag being small is the one piece of evidence against a client-side cause, and it is not a direct measurement of the receive side. This check therefore did not clear the client path and did not isolate the server path. It also does not explain the late first arrival (8.48 s and 9.68 s after `t_start` against 5.8 to 6.75 s), which this data cannot attribute.
Next step if this line is pursued: have the load generator log an absolute send time and log a timestamp at HTTP receipt in the server front end (instrumentation only; needs a plan and a GPU), so the delay can be split into client to front end and front end to scheduler.

## Trigger step 2 result: front-end timestamps on a new pod (option 13, FLAGGED)
Plan: PLAN.md, trigger step 2 (option 13) and its correction. Host `runpod` = `64bba14d4768`, a new pod set up from scratch (setup logs in `results/phase2_runpod_64bba/setup/`; three setup attempts failed before any install, recorded there: the network volume is at its quota, an edit of mine did not apply, and SGLang main needs a Rust toolchain). SGLang main `0e9cff98f5b7`, versions equal to `runpod2`'s main environment; model and 32 adapters equal to the study checksums. The first launch of the runs did nothing (the pod's runner copy predated part M0) and was relaunched after syncing. Pre-check M0: 30 of 30 request ids matched across `recv`, `sent` and `arrive`. Output: `results/phase2_runpod_64bba/trigger2.txt`, raw runs beside it.
Part M: 10 stock 16-slot starts, seed 1, 20 then 40 rps, all 550 requests matched in every window.
- 20 rps: **no slow start in 10** (p95 TTFT 81 to 106 ms). With the earlier rate of 2 in 8, 0 in 10 has a probability of about 6%; the slow state is rarer on this host, or tied to conditions on the earlier hosts. Not established which.
- 20 rps components, medians across starts: client to front end 2.0 to 3.0 ms (p99 7 to 13 ms), front end 1.7 to 2.1 ms, front end to scheduler 4.5 to 5.2 ms (p99 29 to 34 ms).
- 40 rps: the cliff appears in all 10 starts (p95 TTFT 13.4 to 15.4 s). Front end to scheduler rises to a median of 10.8 to 14.8 ms (p99 34 to 45 ms); client to front end stays at 2.0 to 2.8 ms (p99 24 to 35 ms). Unplanned observation: under load, the added delay before the scheduler sees a request is in the hand-off and scheduler loop, not the client or the front end.
Predictions: (a) holds (client to front end median at most 5 ms and p99 at most 50 ms in every 20 rps start). (b) not tested (no slow start). (c) holds: the spread across starts in time from `t_start` to the first server arrival (0.60 s) is all load-generator start-up (time to the first send, 0.60 s). (d) fails narrowly: one start (`M_main-16-10`, 106 ms) is 23% above the median of 86 ms, over the 20% bar; the others are within 15%.
Reading: on this host the client path adds 2 to 3 ms and does not grow with load; the late first arrivals in the earlier slow starts (8.48 s and 9.68 s after `t_start`) are most likely load-generator start-up, given (c), but those two starts were on another host and were not re-measured. The slow-state trigger remains not established; this run could not test it because no slow start occurred.

## Trigger step 3: CPU contention reproduces the slow state (runpod 64bba14d4768, FLAGGED)

Plan: PLAN.md trigger step 3 and its two amendments. Data: `results/phase2_runpod_64bba/` (`N_*`, `P_*`, `Q_*`, `trigger3.txt`, `run.log`).

- Part N (120 busy loops) was stopped after start 1: the load generator itself was starved (first send 81 s late, client to front end median 107 ms). Not evaluated.
- Part P (affinity split) was stopped during its first loaded start: the container has a CPU quota of 17.85 CPUs although `nproc` shows 128, so affinity does not isolate the server. The 5 pinned no-load controls ran; none slow (20 rps p95 79 to 101 ms).
- Part Q (k busy loops, no pinning, 3 rounds of k = 0, 8, 14):

| k | 20 rps p95 TTFT by round (ms) | slow (p95 > 10 s) | hand-off + scheduler median at 20 rps (ms) | throttled periods during 20 rps |
|---|---|---|---|---|
| 0 | 82, 85, 83 | 0 of 3 | 4.4 to 4.6 | 0.6% |
| 8 | 78, 79, 99 | 0 of 3 | 4.1 to 4.4 | 0.6% |
| 14 | 20380, 5038, 19714 | 2 of 3 (third at 5.0 s) | 6.3 to 19.6 | 0.4 to 0.8% |

Predictions:
- (h) holds: no k = 0 start slow; throttled fraction 0.6%.
- (i) first half holds (2 of 3 k = 14 starts slow); second half fails: slow starts were not more throttled (0.4% against 0.6%). The quota-throttling hypothesis is not supported. What reproduces the slow state is CPU contention from 14 busy processes (with the server and load generator, near the 17.85-CPU quota but not throttled), not throttling.
- (j) holds: of the extra median delay in the two slow starts, 86% and 110% is in hand-off and scheduler.
- The validity rule "first send within 1 s of `t_start`" was set wrong: every start, k = 0 included, sends 5.6 to 7.4 s after `t_start` (the gap includes load-generator start-up; step 2 measured only its spread). Applied as written, no start is valid. The other rule (client to front end median at most 5 ms) holds in every start (1.2 to 2.9 ms). I report the result with that rule only and say so. k = 14 starts sent first requests 0.4 to 1.7 s later than k = 0 starts.

What this establishes: on this host, the slow state at 20 rps can be produced on demand (2 of 3 starts, plus one partial at 5 s) by CPU contention that the client survives, and when it occurs the delay builds between the front end and the scheduler. This fits the earlier slow starts on busy shared hosts. It does not yet say which part of the scheduler loop fails to recover; that is the next step (scheduler profiling at k = 14).

## Trigger step 4: the scheduler is CPU-bound; contention makes its work slower, not preempted (runpod 64bba14d4768, FLAGGED)

Plan: PLAN.md trigger step 4. Data: `results/phase2_runpod_64bba/R_*`, `trigger4.txt`. Phase timing inside the scheduler process (py-spy could not attach: no SYS_PTRACE). Overhead check holds (k = 0 median p95 82 ms against 83 ms in part Q).

20 rps window (about 28 s), seconds of wall time per loop phase (CPU time equal to within 0.15 s in every cell):

| start | p95 TTFT | loop busy | ingest | get next batch | run batch | process result | get-next-batch CPU per call | rho with queue | max queue |
|---|---|---|---|---|---|---|---|---|---|
| k0-1 | 82 ms | 86% | 1.73 | 1.76 | 14.76 | 11.19 | 311 us | 0.31 | 2 |
| k0-2 | 82 ms | 86% | 1.79 | 1.73 | 14.66 | 11.31 | 298 us | 0.42 | 2 |
| k0-3 | 84 ms | 85% | 1.74 | 1.77 | 14.86 | 11.06 | 315 us | 0.32 | 1 |
| k14-1 | 7.1 s | 88% | 1.94 | 1.67 | 23.87 | 5.70 | 499 us | 0.82 | 37 |
| k14-2 | 28.3 s | 85% | 3.90 | 3.76 | 39.16 | 3.05 | 2206 us | 0.72 | 132 |
| k14-3 | 9.8 s | 84% | 3.14 | 2.58 | 27.15 | 7.20 | 384 us | 0.77 | 50 |

Predictions (one slow start by the 10 s rule, `k14-2`; the other two k = 14 starts reached 7.1 s and 9.8 s):
- (k) preemption fails: involuntary context switches 4 per second against 6 at k = 0, off-CPU time 0.3%.
- (l) work grows with the queue holds: get-next-batch CPU per call 2206 us against 311 us, Spearman 0.72 with queue length (0.72 to 0.82 in all k = 14 starts, 0.31 to 0.42 at k = 0).
- (m) adapter churn fails: loads per second fell (0.78 against 4.68).
- (n) self-sustaining fails: the queue in `k14-2` went above 50 and later drained.
- Step 2's (b) on `k14-2`: 71% of the extra delay in hand-off and scheduler, below the 80% bar.

What this shows: on a quiet host the scheduler's single thread is already busy about 85% of the 20 rps window, almost all in its own CPU work. Under contention it is not descheduled; the same work takes more CPU time (run batch up 60 to 165%), so demand passes 100% of one thread and the queue grows; then per-call batch selection cost grows with the queue, which slows it further. This is overload of a CPU-bound loop, not a latch.
Limits: CPU time includes any CUDA synchronization spin-wait, so "CPU-bound" in run batch and process result may partly be GPU waiting; why the same work costs more CPU under contention (shared cores' SMT siblings, cache or memory bandwidth) is not measured; one host, three starts per condition.
Things I got wrong: I expected preemption (k); the scheduler thread was not preempted.

## Trigger step 5: batch selection is not the main cost (runpod 64bba14d4768, FLAGGED)

Plan: PLAN.md trigger step 5. Data: `results/phase2_runpod_64bba/S_*`, `trigger5.txt`. 1 start at k = 0, 3 at k = 14; one k = 14 start was slow (p95 16.3 s), two were not (496 ms, 642 ms).
- Batch selection (`get_next_batch_to_run`) costs 50 us of CPU per call at k = 0 and 55 us in the slow start, about 5% of the loop.
- (o) holds by its wording but means little: the increase per call is only 4.9 us, so shares of it are unstable (one share is 201%, another is -207%). The untimed rest of `get_new_batch_prefill` and `can_schedule_lora_req` grow with the queue.
- (p) holds: `can_schedule_lora_req` calls rose from 27 to 1151 per second in the slow start (requests re-checked on each pass while slots are full), at about 2.4 us each.
- Step 4's "2206 us per call" was a median of per-bucket ratios in one start and overstated batch selection's weight; its total share of the loop was small there too (3.76 s of loop wall time, against 39.16 s in `run_batch`).
Exploratory check on parts Q and R traces (not pre-registered): slow and borderline k = 14 starts had 10 to 65% of 20 rps decode steps above batch 48 at 33 to 62 ms each, against about 2 ms at 48 or below; k = 0 starts stayed at 36 or below. Step 6 tests this.
Things I got wrong: I named batch selection the most likely code-fix target after step 4.

## Step 6: falling off the decode graph is the mechanism; eager decode is slow with or without LoRA (runpod 64bba14d4768, FLAGGED)

Plan: PLAN.md step 6. Data: `results/phase2_runpod_64bba/T_*`, `U*`, `step6.txt`.

Part T (14 busy loops, interleaved):

| arm | 20 rps p95 by round | 40 rps p95 | decode steps above 48 at 20 rps |
|---|---|---|---|
| stock (48) | 13.8 s, 0.74 s, 8.1 s | 13.8 to 13.9 s | 32.6%, 0.1%, 22.8% |
| graph 256 | 0.55 s, 0.95 s, 1.57 s | 2.7 to 3.3 s | 0%, 4.6%, 6.9% |

(q) holds. Under contention, the stock starts that went slow are the ones whose decode batches left the graph; with the larger range none reached the stock slow levels, though 40 rps p95 is still about 3 s under contention (against about 14 s stock).

Part U (no busy loops, closed loop), median wall time per decode step in ms:

| server, requests | batch 1 to 48 | 49 to 64 | 65 to 96 |
|---|---|---|---|
| LoRA, stock, adapter requests | 1.7 to 1.9 | 31.5 | 31.4 |
| LoRA, stock, base-model requests | 1.7 to 1.8 | 31.3 to 31.5 | 31.4 |
| LoRA, graph 256, adapter requests | 1.7 to 1.9 | 2.0 | 2.0 |
| no LoRA, stock | 1.3 | 18.7 to 18.9 | 18.7 |
| no LoRA, graph 256 | 1.2 | 1.3 | 1.3 |

- (r) fails: LoRA eager is 1.7 times no-LoRA eager, not 3 times. Eager decode is slow without LoRA too: 18.7 ms against 1.3 ms in the graph (about 14 times), flat across batch 49 to 96.
- (s) fails: LoRA batch preparation is 0.57 ms per step, about 5% of the 12.6 ms LoRA excess.
- Base-model requests on the LoRA server cost the same as adapter requests outside the graph (31.3 against 31.5 ms), so the LoRA excess is in the LoRA-wrapped layers being called, not in adapter weights.
- Flat cost across batch size means eager decode here is bound by per-step CPU work (Python and kernel launches), not GPU work. That is also why CPU contention makes it worse.
Root cause, as far as measured: the scheduler thread is CPU-bound; decode batches above the graph range run eagerly at about 15 to 25 times the per-step cost; any push past 48 (load, or CPU contention slowing the loop) makes steps much longer, so more requests queue and batches stay above 48. Keeping decode in the graph removes most of it.
Things I got wrong: I expected the eager cost to be LoRA-specific (r) and in batch preparation (s); neither held.

## Prior art upstream (searched 2026-11-25, after step 6)

The cliff mechanism was reported before this study's diagnosis. Anything posted from here must cite it.
- sgl-project/sglang#33483 (Aug 2026, auto-closed for inactivity 2026-11-08, not fixed): an L40 with Qwen2.5-0.5B and default flags. Decode graphs are captured to 32 while `max_running_requests` is 4096, set by unrelated code; once the batch crosses the limit, decode runs eagerly and the system settles in a slow state it does not leave (19 of 20 runs), explained by Little's law. Both workarounds were measured there: `--max-running-requests 32`, and `--cuda-graph-max-bs 128`, which was better at the tail. A comment there reproduced it on an RTX 4090 at the 48 default (5.8 times worse median TPOT at 32 rps) and measured the cost of 48 to 128 as about 1.1% of the KV pool and 0.5 s of startup. The issue suggests matching the two defaults, a warning, and capturing larger graphs on first overflow.
- #33900 (warning when a batch crosses the largest graph): never reviewed, auto-closed.
- #37898 (32 GB tier 24 to 48, merged 2026-09-04), #46132 (35 to 60 GiB tier to 128, merged 2026-11-13), #40346 (RTX 5090 to 128, open since 2026-09-19, no review).
- No issue or PR found that proposes running a decode batch above the graph range as several graph replays.
What this study adds, as far as found: the LoRA case (eager decode at 31 ms per step against 19 ms without LoRA at the same batch size); CPU contention as a trigger at a load the server otherwise handles (20 rps); scheduler-loop measurements showing the loop is CPU-bound and is not preempted; and that the A10, 4090 and 5090 tier at tp below 4 is still 48 on main (`0e9cff98f5`).
Things I got wrong: I diagnosed and drafted (`draft-issue.md`) without searching upstream issues for the mechanism first.

## Step 7: cap running requests against a larger graph range (runpod 64bba14d4768, FLAGGED)

Plan: PLAN.md step 7. Data: `results/phase2_runpod_64bba/V_*`, `step7.txt`. 20 rps and 40 rps p95 TTFT, 40 rps output tokens per second, largest decode batch:

| arm | 14 busy loops, round 1 | 14 busy loops, round 2 | no busy loops |
|---|---|---|---|
| stock | 81 ms, 13.6 s, 2995, 244 | 84 ms, 13.5 s, 2976, 244 | **10.1 s**, 15.8 s, 2547, 292 |
| cap48 (`--max-running-requests 48`) | 499 ms, 2.8 s, 4186, 48 | 77 ms, 2.1 s, 4415, 48 | 93 ms, 2.6 s, 4236, 48 |
| cg256 | 2.1 s, 6.8 s, 4607, 231 | 957 ms, 7.1 s, 4192, 156 | 86 ms, 2.2 s, 4632, 82 |

- (t) holds: under contention, cap48 kept 20 rps p95 at or under 499 ms, and its 40 rps p95 (2.1 to 2.8 s) is well below cg256's (6.8 to 7.1 s).
- (u) fails on latency: without contention, cap48 throughput at 40 rps is 8.5% below cg256 (within 10%), but its 40 rps p95 is 2.6 s, above the 1 s bar (cg256: 2.2 s).
- (v) holds: no cap48 decode step above 48.
- The stock start without busy loops went slow at 20 rps (10.1 s, batches up to 292): the slow state does not need contention.
- Under contention the larger range lets batches grow to 156 to 231 and step time rises (median TPOT 13.6 to 17.2 ms against 9.6 for cap48); keeping batches within the graph does better than enlarging the graph. Without contention the two are close, with cg256 slightly ahead, as #33483 also found at the tail.
Reading: keeping every decode step inside a captured graph is what matters. A cap does it at a concurrency cost, which is why vLLM derives its capture range from its running limit (on a 4090: running limit 256, graphs to 512). Next: a fix that keeps decode in the graph without lowering concurrency (derive the range from the limit, or split large batches into graph-sized replays).

## Step 9: split decode into graph replays, against stock, cap, graph 256 and vLLM (runpod 64bba14d4768, FLAGGED; host busier than earlier parts, external load 4 to 9)

Plan: PLAN.md step 9 and its amendment. Data: `results/phase2_runpod_64bba/X0_*`, `X_*`, `Y0_*`, `stepX.txt`.
Part X0 (correctness): split answered all 96 greedy requests; identical completions against cg256: split 65 of 96, stock 61 of 96, cg256 against itself 79 of 96 (batch makeup changes numerics). (z0) holds.

Part X, split version 1 (20 rps p95, 40 rps p95, 40 rps tok/s; rounds 1 and 2):

| arm | no busy loops | 14 busy loops |
|---|---|---|
| stock | 85 ms, 13.5 s, 2986; 82 ms, 13.5 s, 3014 | 10.6 s, 16.9 s, 2354; 821 ms, 16.7 s, 2256 |
| cap48 | 85 ms, 1.76 s, 4524; 83 ms, 1.94 s, 4471 | 737 ms, 7.9 s, 3162; 3.4 s, 13.3 s, 2384 |
| cg256 | 92 ms, 2.2 s, 4664; 102 ms, 6.2 s, 4265 | 340 ms, 7.7 s, 4407; 3.3 s, 9.2 s, 3913 |
| split v1 | 399 ms, 11.5 s, 3652; 4.4 s, 11.1 s, 3783 | 3.8 s, 12.7 s, 3484; 1.1 s, 12.8 s, 3459 |
| vLLM 0.34.0 | 146 ms, 1.54 s, 4646; 153 ms, 1.47 s, 4655 | 949 ms, 2.7 s, 4364; 970 ms, 3.7 s, 3976 |

- Split v1: (z1) fails (oversized steps 12 ms against 1.8 to 2.0 ms), (z2) fails, (z3) fails against cg256. It beats stock but keeps the cliff.
- (z4) for v1: vLLM's 40 rps p95 is lower than split's (holds); throughput within 5% fails (split about 20% lower).
- vLLM has the lowest 40 rps p95 in every start, with and without busy loops.

Part Y0 (closed loop, concurrency 96, sub-phase timing): v1 against v2 (v2 removes a blocking `torch.tensor(n, device=cuda)` per chunk).

| | v1 (2 starts) | v2 (2 starts) |
|---|---|---|
| decode step above 48, median | 12.0 ms | 4.0 ms |
| `_make_chunks` per call | 7.31 ms | 0.21 ms |
| `LoRAManager.prepare_lora_batch` per call | 1.0 ms (chunks plus the parent batch) | 1.0 ms |
| graph `execute` per chunk | 0.64 ms | 0.65 ms |
| output tok/s | 3315, 3319 | 3565, 3716 |

(z5) holds: 67% lower. The blocking copy serialized each chunk's preparation behind the previous replay. Remaining split overhead is mostly LoRA batch preparation, about 1 ms per chunk plus 1 ms for the whole batch, which the split then discards.
Things I got wrong: I expected the per-chunk cost to be attention metadata planning; it was a blocking host-to-device copy.

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
