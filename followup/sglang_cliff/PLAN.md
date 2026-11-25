# SGLang cliff past the LoRA slot limit: Phase 1, is PR #46500 the fix? (plan, written before any run; the time is the commit's)

Source reading and hypotheses: `SOURCE.md` (Phase 0). Author's go for Phase 1 given in the session; threshold 0.2 added with the author's yes.

## Question
Does SGLang `main` (`0e9cff98f5b7`, contains #46500) remove or reduce the latency cliff that v0.5.23 shows with 32 adapters on 16 slots, Zipf traffic, at 40 rps?

## Design
Six server configurations, same model, flags and load as the study's `sgl_flags` runs (`scripts/exploratory/sgl_flags.sh`, control cell):

| label | SGLang | `--lora-drain-wait-threshold` |
|---|---|---|
| v23-def | v0.5.23 (`/root/envs/sglang`) | not passed (0.0, drainer off) |
| v23-d1 | v0.5.23 | 1.0 |
| v23-d02 | v0.5.23 | 0.2 |
| main-def | main `0e9cff98f5b7` (`/root/envs/sglang_main`) | not passed (0.0, drainer off) |
| main-d1 | main | 1.0 |
| main-d02 | main | 0.2 |

Common server flags (study's tuning winner T1, no extra flags): `--model-path /dev/shm/models/Qwen2.5-1.5B-Instruct --served-model-name qwen --mem-fraction-static 0.85 --context-length 4096 --enable-lora --max-lora-rank 16 --max-loras-per-batch 16 --lora-target-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj --lora-paths adapter_0000 ... adapter_0031` from `/dev/shm/adapters` (B scale 0.05).
Adapters: before the first start, the sha256 of `adapter_model.safetensors` of all 32 is checked against `results/raw/adapters_manifest.json`; a mismatch stops the run.
Load: `scripts/loadgen.py` through `scripts/run_ladder.py --workload zipf32 --rates 20,40 --requests 500 --warmup 50`, one repeat per server start, seed `1 + round - 1` (the same prompts and arrival times for every server in a round). Two warm-up runs of 300 requests with no adapters (seeds 998, 997) after each start, as in `sgl_flags.sh`.
Repeats: 3 rounds, each with a fresh server start for every configuration (18 starts). Order within a round, fixed here:
- round 1: v23-def, main-def, v23-d1, main-d1, v23-d02, main-d02
- round 2: main-d02, v23-d02, main-d1, v23-d1, main-def, v23-def (round 1 reversed)
- round 3: v23-d1, main-d02, main-def, v23-d02, main-d1, v23-def
Rounds 1 and 2 are mirror images; round 3 is not balanced against them and I do not claim full balance.
Pass rule (the study's): a run passes if its p95 TTFT is 500 ms or less with 0 errors; a step passes if the median of its runs does (`run_ladder.py`). Here there is one run per start, so the per-repeat pass is the run's own p95. Runs with send lag over 0.05 s are flagged load-generator-limited and reported as such.
Recorded per run: loadgen JSON, server log, `date -u`, host, load average, external CPU (`extload.py check`), GPU state, the effective server args from the log, and for each start whether any other GPU process was present. The run refuses to start a server unless the GPU has no compute process and no `run_rerun.sh`, `run_session.sh`, `run_item*` or `run_cold_pass.sh` process is alive.

## Reading rule (fixed now)
- **Removes the cliff:** main with some drain setting passes the 500 ms bar at 40 rps in all 3 repeats, at a setting where v0.5.23 fails in the repeats it ran (v0.5.23 at the same setting; v23-def is reported alongside).
- **Reduces:** main's median p95 at 40 rps is at least 2 times lower than v0.5.23's at the same setting, with non-overlapping min-max ranges.
- **Otherwise:** no material effect.
- main-def is reported separately, since that is what users get. Its comparison is against v23-def.
- Past-the-cliff p95 values come from a 550-request burst (`SOURCE.md` section d). They are compared only as pass/fail and as the ratio between configurations at the same rate and N, never as steady-state latencies and not across rates.
- The 20 rps cells are the below-cliff control: if any configuration fails 500 ms at 20 rps in a repeat, that cell is flagged and not read as a cliff result.
- Max TTFT, completed counts and errors are reported for every run.

## Shared host
The host is the shared pod `runpod` (hostname must read `126ab78ffbd7`, the one in `followup/pod_items/setup_runpod.sh` and the CHANGELOG; any other hostname stops the run). Other tenants' CPU load is present, so this is a SCREENING run, marked FLAGGED in every table. The expected effect is about 10 times, so a flagged run can still rule a 2 times change in or out; it cannot support a timing claim for the page. The same cells are added to the dedicated-session run list (`followup/dedicated_session/PLAN.md`) as a follow-up amendment if Phase 1 shows an effect.
One GPU job at a time: this run does not start while another study job is running (check above). No engine behavior is changed: stock builds, flags only as listed.

## If main removes the cliff
Stop and report which setting is needed, and whether the default (0.0) leaves users exposed. Phase 2 and 3 are then not run.

## Not claimed
One GPU type, one model (Qwen2.5-1.5B), one traffic shape, 3 repeats, a 550-request burst. A "no material effect" result is for these settings only.

## Amendment 1: run on `runpod2` instead of `runpod` (written before any run; time is the commit's)
The author chose `runpod2` ("2") after I reported that `runpod`'s GPU was in use by other study jobs, and said the author is the only user of `runpod2`. `runpod2` is hostname `48bd787e389e` (RTX 4090, driver 595.112.07), fresh; confirmed against the notes at the start of this setup. Nothing from this plan had run on `runpod`; the waiting runner there was stopped.
Changes: the host rule becomes "hostname must read `48bd787e389e`" (`HOST_EXPECT`); the loadgen interpreter is the v0.5.23 environment's Python; environments, model and the 32 adapters are built fresh on `runpod2` (SGLang v0.5.23 from PyPI; main `0e9cff98f5b7` from source, editable). The adapters are copied byte for byte from `runpod`'s `/dev/shm/adapters` and checked against `results/raw/adapters_manifest.json` before the first start (the runner stops on a mismatch); the model's sha256 is compared with the study's. Torch, flashinfer and kernel package versions resolve at install time and are recorded in `results/phase1/versions.txt`; they may differ from `runpod`'s, which is a limit on comparing these numbers with the earlier `sgl_flags` ones.
The reading rule, matrix, pass rule and order are unchanged. The host is not shared with other tenants as far as the author knows, but I measure load average and external CPU at every start and report them; if they show other load, the run stays FLAGGED. The dedicated-machine claim is not made: the rule in the study is that a timing claim for the page needs the dedicated machine.

## Phase 2: mechanism (plan and predictions, written before any Phase 2 run; the time is the commit's)
Phase 1 result: `results/phase1/analysis.txt`, CHANGELOG entry. #46500 is not the fix, so Phase 2 looks for the mechanism. One repeat per cell is enough for a mechanism and is all that runs; no spread is available, and the results are read as "which hypothesis the data supports", not as precise numbers.

### Instrumentation (log only)
SGLang's own metrics do not give per-adapter loads, batch composition by adapter or queue wait by adapter, so a tracer is used: `trace/cliff_trace.py` plus `trace/sitecustomize.py`, put on `PYTHONPATH`. It wraps `Scheduler._add_request_to_queue`, `Scheduler.run_batch` and `LoRAManager.fetch_new_loras` after import, passes every argument and return value through unchanged, swallows its own errors, and edits no SGLang file (the "separate worktree" is no worktree at all; the SGLang installs stay stock). It writes `arrive` (queue length, running batch size, whether the adapter is resident in the pool and in the running batch), `batch` (mode, requests, distinct adapters, CPU time in the call, gap since the previous call, request ids for prefill batches) and `load` (adapters loaded and evicted, CPU time in the call) events with `time.time()`. Load duration is CPU time in the call; GPU copy time is not synchronized and is not included (the scheduler thread is what stalls on the load, and an unsynchronized call under-counts it if the copy is asynchronous: stated limit).
vLLM: its own `/metrics` (`vllm:request_queue_time_seconds`, `vllm:num_requests_waiting`) read before and after each run; no patch.
Overhead check (before the mechanism runs): main, 16 slots, default flags, 20 rps, Zipf-32, 500 requests plus 50 warm-up: unpatched and patched in the order U P P U U P (one fresh server start each). Tracing is accepted if the median p50 TTFT and the output tokens per second of the patched runs are each within 10% of the unpatched runs'. If not, the mechanism runs are flagged as perturbed by tracing and read with that stated.

### Runs
Mechanism: SGLang v0.5.23 and main `0e9cff98f5b7`, default flags, `--max-loras-per-batch` 16 and 32, Zipf-32, 20 rps then 40 rps on one server start (as Phase 1), seed 1, 500 requests plus 50 warm-up, traced. Four server starts, order: v23-16, main-16, main-32, v23-32. Analysis uses every arrival inside the loadgen window (550 requests including the 50 warm-up; the arrival window is a fixed burst, see `SOURCE.md` d).
vLLM: 0.34.0, `--max-loras 16`, 32 adapters, Zipf-32, 20 and 40 rps, one start, seed 1, same flags as the study's `run_main.sh` (T1).
Same host rule as Amendment 1 (`48bd787e389e`), same busy rule, load and external CPU recorded at every start; the results stay FLAGGED.

### Predictions (fixed now; each is checked in both SGLang versions)
Cells: A = 16 slots 40 rps, B = 32 slots 40 rps, C = 16 slots 20 rps.
- **H1 thrashing:** loads per second in A at least 3, and at least 10 times B's; B has no evictions; loads in B at most 40 in total.
- **H2 fragmentation:** median distinct adapters per decode batch in A at least 14 (at the cap), and mean decode batch size in A at most 0.75 times B's.
- **H3 starvation by adapter, not by position:** in A the median queue wait (arrival to first prefill batch) of requests whose adapter was not resident on arrival is at least 3 s and at least 10 times that of resident ones; Spearman correlation of wait with queue length at arrival below 0.5; in B and C the median wait of both groups is under 0.5 s.
- **H4 synchronous loads stall the scheduler:** summed load time over the busy window (first arrival to last batch) is at least 15% of it in A and at least 10 times B's. Under 5% in A means H4 is not supported.
- **H5 capacity:** output tokens per second over the loadgen window in A at most 0.75 times B's, together with a smaller mean decode batch size (shared with H2).
- **vLLM comparison:** vLLM's mean queue time at 40 rps, 16 slots is at least 5 times lower than SGLang's mean wait in A.
Reading: a hypothesis is "supported" if all its predictions hold in both SGLang versions; "contradicted" if its main observable in A is within 20% of the control (B or C); otherwise "not distinguished". More than one can hold. Numbers that were not predicted are reported as unplanned.

### Output
`results/phase2/` (raw), `results/phase2/analysis.txt`, and `RESULTS.md`: what was measured, which hypothesis the data supports, what is open. Phase 3 (a patch) only after the author's go.

## Phase 3: depth-first slot admission (patch A), screening A/B

### What changes
Patch `lora_depth_first.diff` on SGLang main `0e9cff98f5b7`, one file (`scheduler.py`, +34 -1), applied in a separate worktree `/root/src/sglang-lora-depth` on `runpod2` (host `48bd787e389e`); the stock install and `/root/src/sglang-main` are not edited. The patched server gets the worktree through `PYTHONPATH`; the printed `sglang.__file__` is logged at each patched start. Behavior is off unless `SGLANG_LORA_DEPTH_FIRST=1`.
Rule: when the running set has `f` free adapter slots, a waiting request whose adapter is not running is admitted only if its adapter is among the `f` adapters with the most waiting requests (ties admit). A request that has waited more than `SGLANG_LORA_DEPTH_MAX_AGE_S` (default 10) is always allowed, so a shallow adapter is delayed, not starved. Requests for running adapters are untouched. The drainer stays off (default).
The 10 s cap was chosen before any patched run and is not tuned.

### Runs (part E of `run_phase2.sh`)
Main `0e9cff98f5b7`, 16 slots, traced, Zipf-32, seed 1, 20 rps then 40 rps on one start, six starts in the order U P P U U P (U = stock, P = patched). Same host rule, busy rule and warm-up as Phase 2. FLAGGED screening only; no timing claim for the page.

### Predicted effect (fixed now)
Measured at 40 rps, medians over the three starts of each kind:
- **P1:** patched p95 TTFT is at most 0.5 times unpatched, ranges not overlapping, and still above 500 ms. I do not expect this patch to reach the 500 ms bar, because the demand for distinct adapters at 40 rps is about 31 and 16 slots cannot serve it.
- **P2:** patched mean decode batch size and completed requests per second are each at least 1.15 times unpatched.
- **P3:** patched p99 TTFT is not above unpatched p99 (the age cap stops tail adapters from being made worse).
- **P4 (20 rps):** all three patched starts have p95 TTFT at most 500 ms with 0 errors.
Reading: P1 and P2 hold = "reduces the cliff, does not remove it" and the patch is a fix candidate for the capacity part. Patched p95 within 20% of unpatched = the capacity explanation is contradicted. P3 failing = the patch trades tail latency for throughput and is not a candidate as written.
Not predicted: whether it matches vLLM, and why 20 rps sometimes tips into the slow state.

### Amendment 2 (Phase 3): count how often the rule fires
Reason: the A/B showed no difference and the tracer cannot tell "the rule never rejected anything" from "it rejected and nothing improved". Change: the patch (log-only, no change to any decision) counts calls, calls with no free slot, aged-allow decisions and rejections inside `_lora_depth_ok` and logs them every 2000 calls (`lora_depth calls=... rejected=...` in `server.log`). `lora_depth_first.diff` is updated. Run: part E with `ORDER="P"`, one patched start (tag `E_main-16-P1`), 20 rps then 40 rps, seed 1, same host rule. Prediction: rejections are at least 5% of the calls that reach the depth test (calls minus no-free calls), and the 40 rps result stays within 20% of the stock medians from the A/B (p95 13.5 s, 23.6 requests/s). If rejections are under 1% the earlier no-effect result says nothing about depth-first admission, only that the rule rarely fired. Same FLAGGED screening rules.

### Phase 3 option 3 (chosen by Brian after Amendment 2): slot count as the lever
Reason: reordering (the drainer, depth-first) has nothing to work with when slots are free in 0.1% of calls. The remaining lever in the data is the number of slots, which sets how many distinct adapters run at once.
Runs (part G of `run_phase2.sh`): stock main `0e9cff98f5b7`, `--max-loras-per-batch` 20, 24 and 28, traced, Zipf-32, seed 1, 20 rps then 40 rps, one start each; the 16-slot and 32-slot cells from the A/B and Phase 2 are the references. FLAGGED screening, one repeat per cell.
Predicted (fixed now): S1: the 40 rps run passes (p95 TTFT at most 500 ms, 0 errors) at 28 slots and fails at 20 and 24 slots. S2: completed requests per second at 40 rps rises with slot count (16 < 20 < 24 < 28) and at 28 slots is at least 0.9 times the 32-slot cell (31.9). S3: the share of rule calls with no free slot, not measured here; no prediction.
Reading: S1 and S2 hold = the cliff location follows the number of distinct adapters in use, and the working-set size is the thing to configure. A pass at 24 or a fail at 28 = the threshold is elsewhere, reported as measured.

### Phase 3 option 4: would a reuse predictor pick better drain victims? (offline replay, no GPU)
Idea (Brian): swap out the adapter least likely to be reused. Where it can matter: which running adapter is drained to free a slot (the drainer picks the one with the fewest tokens left). Pool eviction does not matter here (loads are 2 to 4% of busy time).
Method: `replay_slots.py` replays the 40 rps request lists already collected (`zipf32_r40.json`: adapter, send time, time to first token, done time) through a small model of a 16-slot server. A request occupies its adapter's slot from admission for its measured decode time (done minus first token); at most 16 adapters hold slots; waiting requests are admitted FCFS, skipping those whose adapter does not fit (the rule in SGLang). Admission wait plus 0.05 s is the modeled TTFT. Policies: F = FCFS only. R1 = drainer-like: when a non-resident request has waited 1 s, the running adapter that will empty soonest stops taking new requests until it empties (one victim per starving adapter). R2 = same, victim chosen by the smallest expected number of arrivals before it empties, using each adapter's arrival rate over the past 30 s. R3 = victim with the lowest recent arrival rate only. R4 = oracle: victim with the fewest actual arrivals before it empties (a bound; not implementable).
Cells: E_main-16-U1, U5, P2, P3, P6 (normal state) and U4 (slow state), all 40 rps. The model fixes each request's decode time, so effects of batch size on step time are not modeled. FLAGGED, replay of screening data.
Predictions (fixed now): V1 (model check): modeled F p95 TTFT is within 25% of the measured p95 in at least 4 of the 5 normal cells, and mean wait within 25%; if not, no conclusion is drawn. V2: R1 lowers modeled p95 TTFT against F by at least 15% (the measured drainer ran 35% lower). V3: R2 and R3 each lower p95 against R1 by under 10% (median over cells), so the predictor idea is dropped; if R2 or R3 is 20% or more lower, it is worth a real test. V4: the oracle R4 is under 20% lower than R1, which bounds any predictor.

### Phase 3 option 5: is the decode CUDA graph limit the cause? (flag only, no code change)
Observation (traces `E_main-16-U1`): decode calls with 48 or fewer requests take about 1.9 ms of CPU time, calls with 49 or more take about 31.7 ms, flat to 300. The server log shows decode CUDA graphs captured for batch sizes up to 48 only (`bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48]`, 1.98 GB free after capture at `--mem-fraction-static 0.85`). vLLM 0.34.0 at 16 slots and 40 rps: TPOT p50 10 ms, 36.7 requests/s, mean queue 0.24 s; SGLang main: TPOT p50 43 ms, 23.6 requests/s.
Hypothesis H6: decode batches above 48 run without a graph, each step is about 4 times slower, requests hold slots about 5 times longer, and non-running adapters starve.
Runs (part H of `run_phase2.sh`): stock main `0e9cff98f5b7`, 16 slots, traced, the only change `--cuda-graph-max-bs-decode 256`, Zipf-32, seed 1, 20 then 40 rps, three starts (tags `H_main-16-cg256-1..3`). References: the stock starts E U1, U4, U5 and P2, P3, P6 are not reused as patched cells; stock reference = U1, U5 (normal state) and U4 (slow state). Same host rule, FLAGGED screening. If the server fails to start for lack of GPU memory, the failure is reported as the result and a second plan is written before any change to memory settings.
Predictions (fixed now): H1 median TPOT p50 at 40 rps below 20 ms (stock 43 ms). H2 p95 TTFT at 40 rps at least 3 times lower than the stock median 13.46 s (at most 4.5 s). H3 requests/s at 40 rps at least 1.3 times stock (23.6, so at least 30.7). H4 decode calls with 49 or more requests take under 10 ms of CPU time in the trace. H5 20 rps passes (p95 at most 500 ms, 0 errors) in 3 of 3 starts.
Reading: H1 and H2 hold = the graph limit is a main cause of the 16-slot cliff on this GPU and flag setting. H2 fails with p95 above 10 s = H6 contradicted (the eager step time is a symptom of the batch, not the cause). Partial results are reported as measured. Not predicted: whether vLLM is matched.

### Phase 3 option 6: step-time microbench (I) and rerun (J), both on `runpod2`
Brian asked for both on `runpod2` (host `48bd787e389e`). This pod has had external CPU load (up to 47 cores seen at a start), so the results stay FLAGGED unless the external CPU at every start is under 5 cores; each start logs it.
**I, step-time microbench.** Closed loop at concurrency 16, 48 and 96 (300 requests plus 50 warm-up, seed 5, 256 input and 128 output tokens): once on the base model (`--adapters 0`), once on 16 uniform adapters (all running, no slot contention). Servers, one at a time: main 16 slots stock; main 16 slots with `--cuda-graph-max-bs-decode 256`; vLLM 0.34.0 with `--max-loras 16`. This is a step-time comparison at the same batch composition; it is not a kernel profile. A kernel-level profile (torch profiler in each engine) is not part of this run; if the TPOT ratio below holds and I need to know which kernels, I write a separate plan.
Predictions (fixed now), TPOT p50: M1 stock SGLang at N=96 (16 adapters) is at least 2.5 times the raised-limit SGLang, and at N=16 within 15% of it. M2 raised-limit SGLang divided by vLLM with 16 adapters is at least 1.15 at N=48 and at N=96, and for the base model the same ratio is at most 1.15 at N=48. If the base-model ratio is also at least 1.15, the gap is in the base decode step and not in the LoRA path.
**J, rerun.** Stock main `0e9cff98f5b7` (S) against the same with `--cuda-graph-max-bs-decode 256` (C), at 16 and 32 slots, untraced, Zipf-32, seed 1, 20 then 40 rps, 3 rounds in these orders: round 1 S16 C16 S32 C32, round 2 C32 S32 C16 S16, round 3 C16 S32 S16 C32 (12 starts; `J_<cfg>-r<round>`).
Predictions (fixed now): J1 C16 p95 TTFT at 40 rps is at most 4.5 s in each of the 3 starts and at least 3 times lower than the S16 median. J2 S32 and C32 40 rps p95 each at most 500 ms and within 20% of each other (the larger range costs nothing at 32 slots). J3 20 rps passes (p95 at most 500 ms, 0 errors) in 3 of 3 for C16, S32 and C32. J4 C16 requests/s at 40 rps is at least 1.3 times the S16 median.
Order: I, then J, one GPU job at a time, same host rule and busy rule.

### Phase 3 option 7: does adapter swapping explain the remaining gap to vLLM? (step 1: existing traces, no GPU)
Question: with the graph limit raised, SGLang's 40 rps TPOT is 12 to 14 ms while 16 resident adapters give 6.7 to 8.3 ms (microbench). Is the extra time spent around adapter loads?
Method (`swap_cost.py`): for each decode batch in the 40 rps window of `H_main-16-cg256-1..3` (traced, graph limit 256), compute the interval to the next decode batch and split the intervals into those that contain a `load` event (a load happens between the two decode batches) and those that do not. Also report the sum of load call times and the share of busy time, and the median interval of non-load decode steps against the microbench TPOT at the same mean batch size.
Predictions (fixed now): S1 the median decode interval without a load is at most 9 ms (close to the resident microbench), so the 12 to 14 ms TPOT average is not just slower steps. S2 the intervals that contain a load are at least 15 ms longer than those that do not, in the median. S3 total excess time (sum over load-containing intervals of interval minus the non-load median) is at least 8% of the busy window. Reading: S2 and S3 holding = swap cost is a real part of the gap and a GPU-synchronized measurement is worth planning; S1 failing (non-load steps themselves at 12 ms or more) = the gap is in the step under this traffic, not in swapping; S3 under 3% = swap cost does not explain it. FLAGGED, existing screening traces; the load call time is CPU time in the call and a lower bound.

### Phase 3 option 8: overlap loading on top of the raised graph limit (flags only)
Question: does `--enable-lora-overlap-loading` (asynchronous adapter weight loads overlapped with GPU work) remove the 50 to 60 ms decode stall after each load that option 7 found?
Runs (part K of `run_phase2.sh`): stock main `0e9cff98f5b7`, 16 slots, `--cuda-graph-max-bs-decode 256 --enable-lora-overlap-loading`, traced, Zipf-32, seed 1, 20 then 40 rps, three starts (`K_main-16-cg256-ovl-1..3`). References: the three `H_main-16-cg256` starts (traced, same flags without overlap) and the `J_C16` starts (untraced); p95 2.08, 3.14, 2.18 s and 2.15, 2.43, 2.18 s at 40 rps. Same host rule; FLAGGED. The tracer wraps `LoRAManager.fetch_new_loras`; with overlap loading the load may go through a different path, in which case load events may be missing and the swap measure (K4) cannot be computed; that is reported, not worked around.
Predictions (fixed now): K1 median p95 TTFT at 40 rps at most 1.8 s, each start below the reference median 2.18 s. K2 requests/s at 40 rps at least the reference 36.4 in each start. K3 20 rps passes (p95 at most 500 ms, 0 errors) in 3 of 3. K4 (if load events are logged) the excess time in load intervals, computed as in option 7, is at most 12% of the busy window (reference 23 to 24%).
Reading: K1 and K2 hold = overlap loading is the second part of the fix on this setup. K1 fails with p95 within 10% of the reference = the stall is not the loading call path; an error at start or wrong results are reported as measured.

#### Amendment 3 (option 8): launch failed, one flag added
All three starts of part K failed at launch (`ready=0`, before any request), with `AssertionError: Enabling LoRA overlap loading requires pinning LoRA adapter weights in CPU memory, so --max-loaded-loras must be less than or equal to double --max-loras-per-batch: 32`. No measurement was taken. Change: add `--max-loaded-loras 32` (the 32 adapters; 2 x 16 slots, the largest the assertion allows) to the K flags; everything else and all predictions are unchanged. Failed starts are kept as `K_main-16-cg256-ovl-1..3` logs; the rerun uses tags `K2_main-16-cg256-ovl-1..3`.

### Phase 3 option 9: is 48 SGLang's default, or a result of my 0.85? And does the slow state go with batches above 48? (before any issue or page change)
**9.1 Launch-only check (part L of `run_phase2.sh`, `runpod2` host 48bd787e389e; no load, no requests except `/health`).** Stock main `0e9cff98f5b7`, three launches, each stopped after the server reports healthy:
- L1: LoRA as in the study (`--enable-lora --max-lora-rank 16 --max-loras-per-batch 16`, 7 target modules, the 32 adapters, `--context-length 4096`) without `--mem-fraction-static`.
- L2: no LoRA, `--mem-fraction-static 0.85`, `--context-length 4096`.
- L3: no LoRA, no `--mem-fraction-static`, `--context-length 4096`.
Read from each `server.log`: the `Capture target decode CUDA graph end. bs=[...]` line (with free memory), and `mem_fraction_static` and `cuda_graph_max_bs_decode` in `server_args`.
Prediction (fixed now): the largest captured decode batch is 48 in all three launches, so 48 is SGLang's default for a 24 GB GPU and is not caused by my 0.85 or by LoRA. The default memory fraction chosen in L1 and L3 is below 0.85. If any launch captures a different range the page and issue wording change from "default" to the measured conditions.
**9.2 Slow state against batch size (offline, existing traces, `batch_share.py`, no GPU).** Per (start, rate) window of the traced starts (`B_*`, `D_*`, `E_*`, `G_*`, `H_*`, `K2_*`): share of decode calls with more than 48 requests, the mean decode batch size per 1 s bucket, the time of the first request whose TTFT exceeds 1 s, and the longest run of consecutive decode calls above 256.
Slow-state starts: 20 rps with p95 TTFT above 10 s (`B_main-16` at 20, `E_main-16-P1` at 20); 40 rps with throughput of 2.1 to 2.4k tok/s (`B_main-16` and `E_main-16-U4` at 40). Normal: the other starts at the same rate.
Predictions (fixed now): (a) at 20 rps the slow-state starts spend at least 50% of their decode calls above 48, normal starts at most 10%. (b) At 40 rps the two slow-state starts also spend at least 50% above 48, but normal stock starts do too (their mean batch is about 100), so I expect the share not to separate slow from normal there; the author predicted it would, and I will report which of us is right. (c) No start of the raised-limit configurations (`H_*`, `K2_*`) has 20 or more consecutive decode calls above 256. (d) In the slow 20 rps starts the first request over 1 s TTFT comes after the batch has been above 48 for at least 5 s.
Limits: traced starts only (the `J_*` starts are untraced); a decode call's `n` is the batch size at the time of the call; one start per slow case.
