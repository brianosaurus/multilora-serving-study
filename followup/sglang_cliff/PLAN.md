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

### Phase 3 option 10: slot-capture test for the slow state (existing traces, no GPU)
Hypothesis (Brian): in slow starts the 16 slots fill early with adapters that always have at least one running request, so no slot frees and non-resident adapters starve indefinitely.
Method (`slot_capture.py`): per (start, rate) window, rebuild the resident set over time by replaying the `load` events (`loads` added, `evicts` removed) from an empty pool at the start (the warm-up uses the base model only; the replay is checked by requiring the set never to exceed 16). Running requests per adapter over time: a request runs from the first decode-visible batch that holds its rid (admission) until its loadgen completion time; trace arrivals (health checks excluded) are matched to loadgen requests in order, with a time offset taken from the first arrival, and the match is checked by requiring a one-to-one map between trace adapter ids and loadgen adapter names. "After 5 s" means 5 s after the first arrival.
Groups: slow starts = `B_main-16` and `E_main-16-P1` at 20 rps; `B_main-16` and `E_main-16-U4` at 40 rps. Normal = the other stock-limit 16-slot starts at the same rate (`B_v23-16`, `E_main-16-*`). Reported separately: drainer starts (`D_*`) and raised-limit starts (`H_*`, `K2_*`).
Predictions (fixed now), slow starts against normal starts at the same rate:
- (a) after 5 s, the resident set changes (load events carrying at least one load or eviction) at most 2 times in slow starts and at least 10 times in normal starts.
- (b) at least 90% of decode calls after 5 s have every resident adapter holding at least 1 running request in slow starts; at most 50% in normal starts.
- (c) evictions after 5 s are at most 5 in slow starts.
Unplanned, reported: which adapters (popularity rank by request count) held the slots, as a share of slot-time, in each slow start; and whether the drainer starts show slot turnover (load events and evictions after 5 s).
If the replay fails its checks (resident set above 16, or no one-to-one adapter map), no conclusion is drawn and that is the result.
Separately (not a prediction): add one line to `draft-issue.md` on the memory cost of raising the limit, after checking the numbers against the `E_main-16-U1` and `H_main-16-cg256-1` logs.

#### Amendment 4 (option 10): matching method changed after the first run
First run (`results/phase2/slot_capture_v1_positional.txt`, committed with this amendment): the match of trace arrivals to loadgen requests by arrival order failed its own check ("adapter map is not one-to-one") in 24 of 33 windows, because requests sent within milliseconds of each other reach the server in a different order. Under the plan those windows give no conclusion. Change: take each trace adapter id's loadgen name by majority over the order-matched pairs; check the map is one-to-one and that the arrival count per adapter id equals the request count per name; then match the i-th arrival of an adapter to its i-th loadgen request, and take its completion as arrival time plus the loadgen's (completion minus send). Predictions and definitions (a) to (c) are unchanged; no prediction was evaluated against the first run's windows that failed the check. Predictions are stated as written above.

### Phase 3 option 11: queue-aware eviction check (existing traces, no GPU)
**Eviction rule at 16 slots (SGLang main `0e9cff98f5b7`, default `lora_eviction_policy` = `lru`, seen in `server_args` of every start):** `python/sglang/srt/lora/mem_pool.py:810-856`, `get_available_buffer_slot` inside `prepare_lora_batch`. A free pool slot is used first (`:812-814`). Otherwise the candidates are the pool's adapters that are not in the current batch's adapter set (`:821-824`, `cur_uids`) and not pinned (`:827-831`); real adapters are preferred over the base model (`:839-847`); `self.eviction_policy.select_victim(candidates_to_use)` (`:851`) picks the victim, which for LRU is the candidate that appears first in the access order, oldest first (`python/sglang/srt/lora/eviction_policy.py:47`, `select_victim` at `:64-76`). The queue is not consulted anywhere in this path.
Question: at each eviction, did the evicted adapter have requests waiting in the queue?
Method (`evict_queue.py`): per (start, rate) window, at each `load` event with evictions, the waiting count of an evicted adapter at that time = trace arrivals of that adapter (health checks excluded) that arrived before the event and whose rid is not yet in any batch at the event time. Top 4 adapters = the four with the most arrivals in the window. "After 5 s" = 5 s after the first arrival. The evicted adapter id is the trace's `evicts` entry (same ids as the arrivals' `lora`).
Predictions (fixed now): in the slow 20 rps starts (`B_main-16`, `E_main-16-P1`): at least 30% of evictions after 5 s hit an adapter with at least 1 waiting request, and at least one top-4 adapter is evicted with 5 or more waiting. In the normal 20 rps stock starts (`B_v23-16`, `E_main-16-*` not slow): at most 10% of evictions hit an adapter with waiting requests.
Unplanned, reported: the same share at 40 rps, in the drainer starts (`D_*`) and in the raised-limit starts (`H_*`, `K2_*`); and per slow start, the waiting count at each eviction of a top-4 adapter.
If the predictions hold: a plan (not run) for a queue-aware eviction patch, with a predicted effect and a reading rule, written to PLAN.md and left waiting for Brian's go. If they do not hold, no patch plan is written.
Limits: the tracer logs `t` of a load event at the start of the call; a request that arrives during the call counts as arriving after it.

### Trigger step 1 (option 12): is the slow state a load-generator or client-path artifact? (existing data, no GPU)
Question: in the slow 20 rps starts (`B_main-16`, `E_main-16-P1`), was the first request that waited more than 1 s late because of the client side (load generator start-up, HTTP path) or because the server admitted it late? Background: the first arrival in both slow starts came late, measured from `marks.jsonl` `t_start`; the runner takes `t_start` with `date` before the load generator process starts, so that interval includes interpreter start-up and tokenizer load. The load generator's own `max_send_lag_s` is 0.003 to 0.004 s in every 20 rps cell.
Method (`harness_check.py`, reads `results/phase2`): per traced 20 rps window, match trace arrivals to load-generator requests by adapter and per-adapter order (the method of Amendment 4 of option 10, with the same checks; positional matching is not used). For each matched pair: client-to-server delay = trace `arrive` `t` minus the request's `t_send`. Per window report: median and p99 of that delay, the delay over the first 50 requests, `t_send` of the first request minus `t_start`, first server arrival minus the first `t_send`, and for the first request whose admission wait exceeds 1 s: its client-to-server delay and its arrival-to-admission wait.
Predictions (fixed now):
- (a) client-to-server delay has median at most 20 ms and p99 at most 250 ms in every window, slow and normal; in the two slow windows the median is within 2x of the median across the normal windows.
- (b) first server arrival minus the first `t_send` is at most 50 ms in every window; and the extra time to first arrival in the slow windows, compared with the normal windows, comes from `t_send` of the first request minus `t_start` (load-generator start-up), at least 90% of the difference.
- (c) for the first request waiting more than 1 s in the slow windows, its client-to-server delay is at most 250 ms (the wait is on the server, between arrival and admission).
If (a) to (c) hold: the late first arrival is load-generator start-up, not a server effect, and the client path is excluded as the trigger; the trigger is then server-side and the next step is the cold-start A/B (needs a GPU). If any fail, the failing part is reported and no conclusion is drawn about the client path. If the matching checks fail in a window, that window is excluded and said so.
Limits: traced starts only (`J_*` untraced); one start per slow case; the load generator and server run on the same machine, so the two clocks are the same clock.

#### Amendment 5 (option 12): the absolute client-to-server delay cannot be computed; method changed after the first run
First run (`results/phase2/harness_check_v1_absolute_INVALID.txt`, committed with this amendment; script as committed here): predictions (a), (b) and (c) as written compared the trace's server `arrive` time (epoch seconds) with the load generator's `t_send`. In `scripts/loadgen.py`, `t_send` is `time.perf_counter()` minus a zero (`t0`) taken after start-up, so it is relative to the load generator's own start and the records hold no absolute send time. The v1 table therefore shows epoch-sized "delays" and the four "fails" verdicts printed under it are artifacts of that error. They are not a result and say nothing about the client path. The columns that use only server-side events (time to the first request that waited more than 1 s: 3.6 s after the first arrival in `B_main-16`, 3.7 s in `E_main-16-P1`; waits 1.6 s and 2.7 s) are valid and agree with the earlier findings.
What the existing data can test: drift within a run, defined per matched request as (server arrival minus first server arrival) minus (`t_send` minus first `t_send`). A constant client-path delay gives drift near 0; delay that builds up during a run shows as drift growing with time. It cannot give the absolute delay, and it cannot say whether the late first arrival (8.48 s and 9.68 s after `t_start`, against 5.8 to 6.75 s in the normal windows) is load-generator start-up or something else, because the interval from `t_start` to the first send is not recorded.
Predictions for the rerun (fixed now):
- (a') drift has median within 5 ms of 0 and p99 at most 100 ms, in every valid window; in the two slow windows both are inside the range of the normal windows.
- (c') for the first request waiting more than 1 s in the slow windows, its drift is at most 100 ms (the wait is between server arrival and admission, not a growing client-side delay).
Not testable here, to be measured only by a later run after a plan: the absolute client-to-server delay and the load-generator start-up time. That needs the load generator to log an absolute send time (an instrumentation change, not an engine change).
If (a') and (c') hold: the client path adds no delay that grows within a run and the slow windows' first waits are server-side. The late first arrival stays unexplained by this data. If either fails, the failing part is reported.

### Trigger step 2 (option 13): where the slow starts' arrival drift sits (new runs, `runpod`, instrumentation only)
Question: in slow starts, server arrivals fell behind send times (trigger step 1). Is that delay before the HTTP front end (client or network), in the front end (normalizing and tokenizing), or between the front end and the scheduler (inter-process hand-off and the scheduler loop)? And is the late first arrival load-generator start-up?
Host: `runpod`, now host `64bba14d4768`, a new pod. Its `/workspace` holds empty `envs`, `src` and `msf_study` directories, so it is set up from scratch: SGLang main at `0e9cff98f5b7` in `/root/envs/sglang_main` (Python 3.12, uv), the model in `/dev/shm/models` checked against the study's base-weights SHA-256, and the 32 adapters (B scale 0.05) made by `scripts/make_adapters.py` and checked against `results/raw/adapters_manifest.json` (32 of 32 must match, or no run). Versions are recorded and compared with `followup/pod_runpod2/sglang_pip_diff.txt`. The load generator runs in the same environment (`PYL`), not in a separate 0.5.23 environment as before; this is recorded. Different host from every earlier run, so its numbers are compared within this run, not with the earlier pods.
Instrumentation (no engine behavior change): `scripts/loadgen.py` records `t_send_abs` (epoch send time) and `t0_abs`, and with `--rid-prefix` sends a request id `<prefix>-<i>` that SGLang uses as the request's rid, so client, front end and scheduler match exactly. `cliff_trace.py` adds two front-end events: `recv` at `TokenizerManager._init_req_state` and `sent` at `TokenizerManager._send_one_request`, with the rid. A short check before part M: one start, a few requests, confirm that the trace rids equal the sent rids and that `recv`, `sent` and `arrive` all appear.
Runs (part M of `run_phase2.sh`): 10 starts of stock main, 16 slots, traced, same protocol as parts B and E (two warm-up passes, then 20 rps and 40 rps, seed 1). One GPU job at a time; shared host, so FLAGGED.
Per request: client to front end = `recv` minus `t_send_abs`; front end = `sent` minus `recv`; hand-off and scheduler = `arrive` minus `sent`. Slow start: 20 rps p95 TTFT above 10 s, as before.
Predictions (fixed now):
- (a) client to front end has median at most 5 ms and p99 at most 50 ms in every start, slow or not.
- (b) if any slow start occurs, at least 80% of its extra drift, compared with the median normal start, is in hand-off and scheduler (`arrive` minus `sent`).
- (c) the time from `t_start` to the first send (`t_send_abs` of request 0 minus `t_start`) accounts for at least 90% of the variation across starts in time from `t_start` to the first server arrival.
- (d) normal starts' 20 rps p95 TTFT is within 20% of their median across this run (the instrumentation does not change behavior in a visible way).
If no slow start occurs in 10 starts: (b) is not tested, the components are reported for normal starts, and nothing is concluded about the trigger. The slow-state rate in the earlier stock 16-slot traced starts at seed 1 was 2 of 6 starts.
Limits: one host, one seed (as before), front-end and scheduler clocks are the same machine clock; `recv` is after request parsing, so time inside the HTTP server before it is counted as client to front end.
Correction to the line above (before any run): "2 of 6 starts" is wrong. The earlier traced main 16-slot starts with the stock graph limit and no drainer, at seed 1, are `B_main-16`, `E_main-16-U1`, `-U4`, `-U5` (stock code) and `E_main-16-P1`, `-P2`, `-P3`, `-P6` (depth-first patch, which had no measured effect). At 20 rps 2 of these 8 were slow (`B_main-16`, stock; `E_main-16-P1`, patched), and 1 of the 4 stock-code starts.

### Trigger step 3 (option 14): does background CPU load cause the slow state? (new runs, `runpod` host `64bba14d4768`, no engine change)
Why: both earlier slow starts were on shared hosts with other tenants; trigger step 2 on this pod (load average under 1) had 0 slow starts in 10. Under load the added wait was between the front end and the scheduler, which is CPU-bound Python.
Runs (part N of `run_phase2.sh`): same as part M (stock main, 16 slots, traced with rids, two warm-ups, 20 and 40 rps, seed 1), 10 starts, with 120 busy-loop Python processes (`while True: pass`, the host has 128 CPUs) started before each server start and killed after each server stop. The engine and its flags are unchanged. Load average is logged per start. One GPU job at a time; shared host, FLAGGED.
Predictions (fixed now):
- (e) at least 2 of 10 starts are slow (20 rps p95 TTFT above 10 s). If 0 or 1, CPU contention of this kind is not shown to cause the slow state.
- (f) if any start is slow, (b) from step 2 is tested on it: at least 80% of its extra drift over the median normal start in this part is in hand-off and scheduler.
- (g) in normal starts, the hand-off and scheduler median at 20 rps is higher than in part M (4.5 to 5.2 ms).
Limits: busy loops are not the same as the earlier hosts' unknown load; a negative result does not rule out other contention (memory bandwidth, PCIe, disk).

Amendment to trigger step 3 (after start 1 of part N, before any further run): part N is stopped. Start 1 (`N_main-16-cpu-1`) had 20 rps p95 TTFT 84.8 s, but the load generator sent its first request 81 s after `t_start` and client to front end had median 107 ms (part M: 2 to 3 ms). The busy loops starved the client, so part N cannot separate a server slow state from a slow client; predictions (e) to (g) are not evaluated on it. Start 2 (`N_main-16-cpu-2`) was cut off during server start and is incomplete. Both are kept and labeled.
Replacement, part P: the same runs, with CPU affinity separating the two sides. `run_phase2.sh P` itself runs under `taskset -c 0-15`, so the server, its child processes and the load generator use cores 0 to 15; the 120 busy loops are started with `taskset -c 16-127`. They still share memory bandwidth, caches and PCIe with the server, but not cores. Order: 5 control starts (pinned, no busy loops; `P_main-16-pin-1..5`), then 10 loaded starts (`P_main-16-pinload-1..10`). Engine and flags unchanged. FLAGGED.
Validity check per loaded start (fixed now): first send at most 1 s after `t_start` and client to front end median at most 5 ms at 20 rps. A start that fails it is reported as client-starved and not counted.
Predictions (fixed now):
- (e') at least 2 of the valid loaded starts are slow (20 rps p95 TTFT above 10 s), and at most 1 of the 5 controls is.
- (f') in any slow valid start, at least 80% of its extra drift over the median normal start in its group is in hand-off and scheduler.
- (g') in normal valid loaded starts, the hand-off and scheduler median at 20 rps is higher than in the controls.
If (e') fails, contention from other cores is not shown to trigger the slow state, and the next step is in-process profiling of the scheduler (py-spy) during a slow start, which needs its own plan.

Amendment 2 to trigger step 3 (after 5 controls and part of one loaded start of part P, before any further run): part P is stopped. The container has a CFS CPU quota of 17.85 CPUs (`cpu.cfs_quota_us` 1785000, period 100000) although `nproc` shows 128. Affinity does not separate the busy loops from the server under a shared quota: the 120 loops use up the quota and every process in the container is throttled. That also explains part N's starved client. In `P_main-16-pinload-1` the server took 8.5 minutes to start (the controls took about 40 s); it produced no load results. The 5 controls (`P_main-16-pin-1..5`) completed: 20 rps p95 79 to 101 ms, 40 rps 13.4 to 14.6 s, none slow. They are kept as pinned no-load controls.
New hypothesis: the slow state is triggered by CPU quota throttling of the SGLang processes. The earlier slow starts were on shared containers whose quota use was not recorded.
Part Q (replaces the loaded half of part P): no pinning. Busy-loop count k in {0, 8, 14}, chosen to leave the quota unused, near full and over full with the server's own use. Order k = 0, 8, 14, repeated 3 times (9 starts), otherwise as part M. `/sys/fs/cgroup/cpu/cpu.stat` (nr_periods, nr_throttled, throttled_time) is logged at server start, before and after each load rate. Engine and flags unchanged. FLAGGED.
Validity check per start, as in part P (first send at most 1 s after `t_start`, client to front end median at most 5 ms at 20 rps). Starts that fail are reported as client-starved and not counted.
Predictions (fixed now):
- (h) no start at k = 0 is slow, and the throttled fraction (nr_throttled delta over nr_periods delta) during its 20 rps load is under 1%.
- (i) if throttling triggers the slow state: at least 2 of the 3 valid k = 14 starts are slow, and across all valid starts, slow starts have a higher throttled fraction during 20 rps than every normal start.
- (j) in any slow valid start, at least 80% of its extra drift over the median normal k = 0 start is in hand-off and scheduler.
If (i) fails, throttling of this kind is not shown to be the trigger; the next step is py-spy profiling of the scheduler under load (own plan).

### Trigger step 4 (option 15): what in the scheduler loop fails to recover under CPU contention (runpod 64bba14d4768, instrumentation only)
Why: step 3 part Q produced the slow state in 2 of 3 starts with 14 busy loops and none with 0 or 8; the delay is between the front end and the scheduler. py-spy cannot attach (the container lacks SYS_PTRACE), so timing is taken inside the scheduler process.
Instrumentation (no engine behavior change; on only with `SGLANG_CLIFF_PHASES=1`): `cliff_trace.py` wraps the scheduler loop phases `ingest_requests`, `get_next_batch_to_run`, `run_batch` and `process_batch_result`, and writes per 1 s bucket: calls, wall time and thread CPU time per phase, the scheduler thread's voluntary and involuntary context switches, waiting-queue length and running-batch size.
Runs (part R): k in {0, 14} busy loops, order 0, 14, repeated 3 times (6 starts), otherwise as part Q (stock main, 16 slots, traced, rids, cpu.stat logged). FLAGGED.
Overhead check (fixed now): k = 0 starts with phase timing have 20 rps p95 TTFT within 20% of part Q's k = 0 median (83 ms). If not, phase results are reported as perturbed.
Validity: client to front end median at most 5 ms at 20 rps (the first-send rule is dropped; see step 3 result).
Predictions, over the 20 rps window, slow k = 14 starts against k = 0 starts (fixed now; each is one candidate mechanism, more than one may hold):
- (k) preemption: the scheduler thread's involuntary context switches per second are at least 5 times the k = 0 median, and off-CPU time (wall minus CPU, summed over the four phases) is at least 30% of loop wall time, against under 10% at k = 0.
- (l) work grows with the queue: CPU time per `get_next_batch_to_run` call has Spearman at least 0.5 with waiting-queue length across 1 s buckets, and its median per-call CPU time is at least 3 times the k = 0 median.
- (m) adapter churn: adapter loads per second at 20 rps are at least 3 times the k = 0 median.
- (n) the slow state is self-sustaining: after the first 1 s bucket with waiting queue above 50, the queue stays above 50 for the rest of the 20 rps window in slow starts.
If no k = 14 start is slow, the phase data from loaded normal starts is reported and nothing is concluded about the mechanism.

### Trigger step 5 (option 16): which part of batch selection grows with the queue (runpod 64bba14d4768, instrumentation only)
Why: step 4 found get-next-batch CPU per call 2206 us against 311 us, Spearman 0.72 with queue length, under 14 busy loops. Settings in those runs: `schedule_policy` fcfs, no drainer, no overlap loading.
Instrumentation (on only with `SGLANG_CLIFF_PHASES=1` and `SGLANG_CLIFF_SUBPHASES=1`, no behavior change): the step 4 phase buckets also time `Scheduler.get_new_batch_prefill`, `update_running_batch`, `can_schedule_lora_req`, `process_pending_chunked_abort`, `_process_hicache_events`, `SchedulePolicy.calc_priority`, `PrefillAdder.add_one_req` and the waiting queue's `reorder`. Nested calls are counted inside their callers too.
Runs (part S): k = 0 once, then k = 14 three times; otherwise as part R. A pre-check (part S0) first confirms the new names appear. FLAGGED.
Predictions (fixed now):
- (o) in the slowest k = 14 start, one timed function (excluding its callers) accounts for at least 50% of the increase in get-next-batch CPU per call over the k = 0 start, and its CPU per call has Spearman at least 0.5 with queue length.
- (p) `can_schedule_lora_req` calls per second at 20 rps are at least 5 times the k = 0 rate (requests re-checked on each pass while adapter slots are full).
No patch is written or run in this step. A fix is planned in a separate step, against what (o) and (p) find.

### Step 6 (option 17): does the graph range remove the contention trigger, and why is decode outside the graph so slow with LoRA (runpod 64bba14d4768)
Exploratory finding that motivates this (existing traces, not pre-registered): in parts Q and R, slow and borderline k = 14 starts had 10 to 65% of 20 rps decode steps above batch 48 (the decode graph range), at 33 to 62 ms per step, against 1.9 to 3.1 ms for steps at 48 or below; k = 0 starts never went above 36. Part S's first k = 14 start showed batch selection at about 55 us per call (50 us at k = 0), about 5% of the loop: not the main cost.
Experiment 1 (part T, engine flag change by design): 14 busy loops; arms stock and `--cuda-graph-max-bs-decode 256`, interleaved, 3 rounds (6 starts); otherwise as part R. FLAGGED.
- (q) at most 1 of 3 cg256 starts has 20 rps p95 TTFT above 1 s, and every stock start is above 1 s or has decode steps above 48. If (q) holds, falling off the decode graph is what turns contention into the slow state.
Experiment 2 (part U, instrumentation only): no busy loops; closed-loop load at concurrency 32, 64 and 96 (400 requests, 40 warm-up, seed 1). Servers: U1 LoRA stock (adapter requests, then base-model requests), U2 LoRA graph 256 (same), U3 no-LoRA stock, U4 no-LoRA graph 256. Per decode step: `run_batch` wall time by batch size from the trace, plus CPU time in `LoRAManager.prepare_lora_batch` and `ChunkedSgmvLoRABackend.prepare_lora_batch` per step.
- (r) for decode steps of batch 49 to 96: LoRA eager (U1, adapter requests) per-step time is at least 3 times no-LoRA eager (U3) at the same batch size. If so, the extra cost is LoRA-specific.
- (s) of the LoRA eager excess over no-LoRA eager (U1 minus U3), at least 50% is CPU time in the two `prepare_lora_batch` functions. If not, the excess is in the forward pass itself (kernel launches or GPU work), to be split in a later step.
- Reported, no prediction: U1 base-model requests on the LoRA server against U3, and graph against eager per batch size.
A code change is planned only after this step, against what (r) and (s) find.

### Step 7 (option 18): fix candidates for decode outside the graph (runpod 64bba14d4768; engine flags change by design)
Facts from the code (SGLang main `0e9cff98f5`): `DecodeCudaGraphRunner.can_run_graph` sends any decode batch above the largest captured size to eager; nothing splits a large batch into graph-sized replays; `max_running_requests` is unset (None) in these runs and the running batch reached 342.
Two fix candidates:
- A, cap: keep the running batch within the graph range (`--max-running-requests 48`). Testable with a flag. Cost: throughput when an eager large batch would beat a graph small batch (likely on larger GPUs or models; not measured here).
- B, split: run a decode batch above the graph range as several graph replays. Needs a code change in the model runner; planned separately only if A shows that keeping decode in the graph is enough.
Part V: arms stock, cap48 (`--max-running-requests 48`), cg256 (`--cuda-graph-max-bs-decode 256`); with 14 busy loops, 2 rounds interleaved (6 starts), then without busy loops one start per arm (3 starts); otherwise as part T. FLAGGED.
Predictions (fixed now):
- (t) with 14 busy loops, every cap48 start has 20 rps p95 TTFT at most 1 s, and 40 rps p95 at most the cg256 median in the same round set.
- (u) without busy loops, cap48 output tokens per second at 40 rps is within 10% of cg256, and its 40 rps p95 is at most 1 s.
- (v) cap48 has no decode step above 48 in any start.
If (t) to (v) hold, A is a valid workaround on this hardware, and B (the general fix) is planned next. If (u) fails, capping costs throughput even here, and B is needed.

### Step 8 (option 19): a patch that keeps the default running limit within the decode graph range (runpod 64bba14d4768; engine behavior change by design)
Patch (`graphcap.diff`; local branch `decode-graph-running-cap` on SGLang `0e9cff98f5`, not pushed): in `KVCacheConfigurator.resolve_max_num_reqs`, when `--max-running-requests` is not set, the derived limit is capped at the largest decode CUDA graph batch size (skipped when decode graphs are disabled and on PD prefill servers). An explicit `--max-running-requests` is unchanged. This is direction 1 of sgl-project/sglang#33483 and matches how vLLM ties its capture range to `max_num_seqs`.
Patched code is loaded by putting a patched worktree's `python/` first on PYTHONPATH (as for the depth-first patch). A pre-check (part W0) confirms the patched server logs the new default and runs.
Part W: arms stock (no flags), patch (no flags), patch + `--cuda-graph-max-bs-decode 256` (running limit then defaults to 256); conditions k = 0 and k = 14 busy loops; 2 rounds; order within a round: stock, patch, patch-cg256 at k = 0, then the same at k = 14 (12 starts). Otherwise as part V. FLAGGED.
Predictions (fixed now):
- (w) every patch and patch-cg256 start has no decode step above its graph range (48 and 256).
- (x) at k = 0 and k = 14, every patch start has 20 rps p95 TTFT at most 1 s, and its 40 rps p95 is below every stock start's in the same condition.
- (y) at k = 0, patch 40 rps output tokens per second is at least stock's in every round.
- Reported, no prediction: patch against patch-cg256 (the cost of the 48 default against a larger graph range).
Limits: one model (Qwen2.5-1.5B), one GPU, rank-16 adapters at 16 slots; larger models, where eager decode costs less relative to graphs, may lose throughput from the cap, and are not measured here.

Step 8 stopped (before any further run): part W was stopped after 2 starts (`W_main-16-stock-k0-1`, `W_main-16-patch-k0-1`) because the cap patch will not be proposed; its comparisons move into step 9. The two finished starts are kept and labeled, not analyzed.

### Step 9 (option 20): split decode batches above the graph range into graph replays; compare with stock, the cap, graph 256 and vLLM (runpod 64bba14d4768; engine behavior change by design)
Patch (`split.diff`; local SGLang branch `decode-graph-split` on `0e9cff98f5`, not pushed): in `ModelRunner._forward_raw`, a plain decode batch larger than the largest captured graph is cut into even chunks of at most that size (`TboForwardBatchPreparer.filter_batch`), each chunk refills the LoRA batch info and replays a graph, outputs are copied out of the static buffers and concatenated, and sampling runs once on the whole batch. Falls back to eager for speculative decoding, DP attention gathers, dLLM, mrope, PP, and outputs it cannot merge. Loaded by putting the patched worktree's `python/` first on PYTHONPATH.
Part X0 (correctness and smoke, no busy loops): servers stock, split and graph 256 (`cg256`), plus cg256 a second time; 96 concurrent greedy requests (64 tokens, adapters round-robin), completions compared pairwise; then on the split server a closed-loop load at concurrency 96 with phase timing.
- (z0) the split server answers all 96 requests, and its completions match cg256 at least as often as stock's do. Reported, no prediction: cg256 against itself (run-to-run determinism).
Part X: arms stock, cap48 (`--max-running-requests 48`), cg256, split, vLLM 0.34.0 (same flags as part C); conditions k = 0 and k = 14 busy loops; 2 rounds; arm order within each condition as listed. Otherwise as part V. FLAGGED. vLLM is installed in `/root/envs/vllm` with uv; the install log is kept.
Predictions (fixed now):
- (z1) split: decode steps above batch 48 have median wall time at most 3 times the median of steps at 48 or below, in every split start (stock: about 16 times).
- (z2) split: 20 rps p95 TTFT at most 1 s in every start, and 40 rps p95 below every stock start in the same condition.
- (z3) at k = 0, split 40 rps output tokens per second is at least stock's and at least 0.95 times cg256's, in every round.
- (z4) at k = 0, vLLM's 40 rps p95 TTFT is lower than split's (adapter-load stalls, earlier results), and split's throughput is within 5% of vLLM's.
Limits: one model (Qwen2.5-1.5B), one GPU, 16 adapter slots; shared host.

Step 9 amendment (during part X, before any of the runs below): part X0 showed split decode steps above 48 at 12.1 ms median (2 chunks; about 6 ms per chunk) against 2.0 ms for one graph step, so (z1) is expected to fail for the version in part X. Version 2 of the patch (`split_v2.diff`, branch commit `7a32fc283e`) replaces a blocking `torch.tensor(n, device=cuda)` per chunk with `torch.full`, which may have serialized each chunk's preparation behind the previous replay. Part X continues unchanged with version 1.
Part Y0 (after part X; no busy loops): split v1 and v2 alternating, 2 starts each, closed-loop concurrency 96 (400 requests, 40 warm-up), with phase and sub-phase timing, which now also times `DecodeCudaGraphRunner.execute` and `load_batch` and the split helpers (instrumentation only, active only with `SGLANG_CLIFF_SUBPHASES=1`).
- (z5) v2's median wall time per decode step above 48 is at least 30% lower than v1's. Reported: where each chunk's time goes.
Part Y (only if z5 holds): split v2 rerun in part X's protocol (k = 0 and 14, 2 rounds), compared against part X's other arms; (z1) to (z4) are evaluated for v2 as well as v1.
