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
