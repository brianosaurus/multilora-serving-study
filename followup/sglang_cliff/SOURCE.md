# SGLang latency cliff past the LoRA slot limit: Phase 0 (reading only, no GPU, no pod)

Commits read: SGLang `v0.5.23` and `main` `0e9cff98f5b7` (local clone `~/src/sglang`; `0e9cff98f5b7` and the PR commit `259f0131c9` are both in it). vLLM: only tag `v0.34.0rc0` exists in the local clone, not `v0.34.0`, so the vLLM citations below are to the rc0 tag; the study ran 0.34.0. Not yet diffed against the 0.34.0 release.
Citations are `file:line` at `v0.5.23` / at `main`. Nothing here is measured.

## a. What PR #46500 changes

PR: "[Feat] LoRA: revamp LoRADrainer for better performance" (merge commit `259f0131c9`, the PR's own text says the feature is off by default). 300 additions, 241 deletions. Files: `lora_drainer.py` (rewrite), `scheduler.py` (+11/−1), the arg help text in `arg_groups/fields/lora.py`, two docs lines, tests.

**At `lora_drain_wait_threshold = 0.0` it changes nothing it can reach.** The drainer object is only created when the threshold is above zero: `scheduler.py:2400-2407` (v0.5.23), `scheduler.py:2596-2603` (main). Every call into the drainer is behind `if self.lora_drainer` (`scheduler.py:4193` / `4763`, `can_schedule_lora_req` `4426` / `5089`). So with the default, the drainer is off at both commits. Any difference between v0.5.23 and main at the default comes from the other changes between the tags, not from #46500. Those are large: `scheduler.py` +2605/−, `schedule_policy.py` 1340 lines, `mem_pool.py`, `lora_manager.py`, `lora_registry.py` (26 LoRA/scheduler files changed). I have not read them all; the ones on the admission and load path (below) look the same.

What the drainer does when on (main, `lora_drainer.py`):
- An adapter is "starving" once its oldest waiting request has waited longer than the threshold (`:104-121`).
- Draining only starts when running adapters (plus pinned) fill the slots (`:82`). For each starving adapter, longest wait first, it picks the running adapter whose longest request has the fewest tokens left (`:158-161`) and marks it draining.
- A drained adapter accepts a new request only if `max_new_tokens <= anchor request's remaining tokens` (`:102`). Old code allowed 1.2x that (`DRAIN_SCHEDULE_TOLERANCE`, v0.5.23 `lora_drainer.py:25, 113-116`).
- A drain ends when the drained adapter has left the batch, the starving one got a slot, or the starving one has nothing waiting (`:72-79`). Old code only ended a drain when the drained adapter left (v0.5.23 `:160-175`).
- New: counts requests already in `adder.can_run_list`, skips pinned adapters (`scheduler.py` main `:4763-4780`).
- The new help text says draining is "best effort: freed slots are not reserved and waiting time is not bounded". The old text claimed it mitigates extreme tail latency.

PR's own evidence: 4 adapters on 3 slots, uniform, 4 rps, H200, Qwen3-30B-A3B. P99 TTFT 89.6 s to 14.4 s, mean TTFT −70%, median TTFT +93% (77 to 149 ms). That is a small, uniform case at a low rate.

**What it predicts for our workload (32 adapters, 16 slots, Zipf, 40 rps, 256 in / 128 out, `ignore_eos`): a prediction, not a result.** The study's workload fits the PR's stated scenario (more busy adapters than slots, hot adapters keep slots). Every request has `max_new_tokens = 128` (`loadgen.py` sends `max_tokens` and `ignore_eos`). With the strict rule, a drained adapter takes a new request only while its anchor request still has all 128 tokens left, so a drained adapter stops admitting almost at once. The old 1.2x rule admitted new requests while the anchor had at least 107 tokens left. So the new drainer may free slots faster than the old one did; the earlier `drain1` run (old drainer, 1 s) gave −16% p95 only. The size of the effect is what Phase 1 measures.

## b. One request, arrival to first token, adapter not loaded

Same code at both commits unless noted; the line numbers move (v0.5.23 / main).
1. The request enters `waiting_queue` (default `--schedule-policy fcfs`, `arg_groups/fields/schedule.py:86-96`). It records `wait_queue_entry_time`.
2. Each prefill pass, `get_new_batch_prefill` builds `running_loras` from the running decode batch plus already-chosen requests (`scheduler.py:4187-4191` / `4755-4760`). Then it walks the queue.
3. For a LoRA request it calls `can_schedule_lora_req` (`:4209` / `4818`; defined `:4416-4437` / `5079-5100`). That returns True if the adapter is already running. Otherwise it asks `validate_lora_batch({req.lora_id} | running_loras)` (`lora_manager.py:427-456` / `422-451`). That only counts distinct adapters: True if the count is at most `max_loras_per_batch` (after pinned adapters). Whether the adapter's weights are resident is not asked.
4. If it fails, the request is skipped with `continue` (`:4209-4211` / `4818-4820`). It stays in the queue in its original place. Later requests whose adapter is running, or that still fit, are admitted past it. There is no hard head-of-line block in admission. There is also no aging, priority or reservation: nothing makes a skipped request more likely to win later, except the drainer when on.
5. A request is admitted only when the number of distinct adapters in (running decode batch + this prefill batch) is below the slot count. With 16 slots and requests that run ~128 decode steps, the running batch under Zipf can hold 16 distinct adapters most of the time. A request for a 17th adapter then waits until a running adapter has no request left in the batch.
6. Load and eviction happen later, in the forward path, not in the scheduler's admission step: `forward_batch_info.py:1294` / `1291` calls `fetch_new_loras(set(batch.lora_ids))` right before the batch runs (non-overlap mode), then `mem_pool.prepare_lora_batch` (`mem_pool.py:780-862` / `798-880`).
7. Slot choice: empty slot first; otherwise evict one resident adapter not in this batch, not pinned, preferring real adapters over the base model, picked by `--lora-eviction-policy` (default LRU) (`mem_pool.py:792-838`).
8. The load is synchronous on the scheduler/worker thread: `load_lora_weight_to_buffer` copies the adapter host to device. In non-overlap mode the copy source is `weight.pin_memory()` on every load, with no cache (`mem_pool.py:751-779`, the pinned-weight store is only used when overlap loading is on). `--enable-lora-overlap-loading` loads one adapter at a time ahead of the batch instead (`lora_overlap_loader.py:93`).
9. The first token comes from that prefill batch's forward.

Main differs on this path only by: `pp_loras` argument for pipeline parallel (`lora_manager.py:453-460`), and refactors. Admission, slot selection and eviction read the same as v0.5.23.

## c. vLLM, same case

`vllm/v1/core/sched/scheduler.py` at `v0.34.0rc0`:
- `scheduled_loras` = adapters of all running requests (`:1054-1063`).
- For each waiting request, if the set is full (`== max_loras`) and the request's adapter is not in it, it is skipped (`skip_request`, `continue`) (`:1130-1141`). Waiting-queue order is kept.
- The set grows as requests are admitted (`:1628-1629`).

Same rule as SGLang: skip when full, keep order, admit later requests whose adapter is running. The admission rule is not where they differ. Loads/activation are in `worker_manager.py` / `model_manager.py` (`set_active_adapters`, `activate_adapter`); I have not read them in detail yet.

Measured (study data, `results/raw/main_rerun1` for SGLang, `results/raw/main` for vLLM, 3 repeats, Zipf-32, 16 slots, 500 measured requests):

| rate | SGLang p95 TTFT (ms) | SGLang tok/s | vLLM p95 TTFT (ms) | vLLM tok/s |
|---|---|---|---|---|
| 20 rps | 84, 163, 73 | 2583, 2706, 2364 | 50, 72, 48 | 2567, 2683, 2345 |
| 40 rps | 13517, 12142, 11841 | 3056, 3126, 2921 | 1456, 1565, 958 | 4659, 4883, 4319 |
| 80 rps | 9656, 9560, 9441 | 4188, 4220, 4067 | 5629, 4663, 3316 | 7097, 7385, 6842 |

Offered output load at 40 rps is 40 × 128 = 5120 tok/s. SGLang finishes at 2900–3100 tok/s, vLLM at 4300–4900. At 20 rps (2560 offered) both keep up. vLLM also queues badly at 80 rps (3–6 s). SGLang with 32 slots reaches ~3800–4200 tok/s at 40 rps (control run), so SGLang's low throughput is tied to the slot limit.

## d. The 80 rps vs 40 rps clue

SGLang p95 at 80 rps (9.6 s) is below 40 rps (12.1 s). Checked in `results/raw/main_rerun1/sglang/` (3 repeats per rate):

| | 40 rps | 80 rps |
|---|---|---|
| requests per run | 500 measured + 50 warm-up | same |
| completed / errors | 500 / 0 in all 3 repeats | same |
| last request sent at (s) | 12.5 to 14.4 | 6.3 to 7.2 |
| run ends (last completion, s) | 21.5 to 23.1 | 15.7 to 16.3 |
| max TTFT (s) | 15.0 to 15.6 | 10.8 to 10.9 |

How the load generator ends a run (`scripts/loadgen.py`): it schedules all `warmup + requests` arrivals at fixed Poisson times, awaits all of them (`asyncio.gather`), and ends when the last one completes. There is no time limit on the run and no early cut-off. The per-request timeout is 180 s (`REQUEST_TIMEOUT_S`), and nothing came near it (max TTFT 15.6 s), so no request was dropped and no error was counted.

**Yes, the run window caps the queueing we can see.** The number of requests is fixed (550), not the duration. At 80 rps all arrivals are done in about 7 s; at 40 rps in about 13 s. A backlog can only build for as long as arrivals continue, then it drains with no new load. So the longest wait is bounded by the size of the one-off burst (≈ 7 s and 13 s of traffic), not by a steady state. That fits 80 rps having the lower p95 and lower max TTFT. At 80 rps a larger share of requests are slow (29% above 1 s against 22% at 40, control run rep 0), each waiting less.

Consequences for how every past-the-cliff number is read:
- SGLang p95 at 40 and 80 rps (and the 11.8–13.8 s and 9.7–9.8 s quoted in the drafts) are "p95 of a 550-request burst that started from empty", not a steady-state latency. A longer run at the same rate would likely show higher values; I have not run one.
- The ordering of the cliff rates is partly set by N. Past-the-cliff numbers should not be compared across rates, only pass/fail against the 500 ms bar at each rate.
- Throughput numbers past the cliff include the drain tail (last_done is much later than last_send), so they understate the engine's rate while busy.
- The rate ladder pass/fail result (cliff between 20 and 40 rps) does not depend on this, since 20 rps stays under 0.2 s max TTFT.

Another read of the same data (one repeat, control rep 0): at 40 rps, 71% of requests for the 16 least popular adapters (ranks 17–32) wait more than 1 s, against 4% for the 8 most popular. At 20 rps nobody does. That is consistent with starvation of tail adapters but does not distinguish the hypotheses below.

## e. Hypotheses and what tells them apart

Predictions are stated before any run (Phase 2 will restate them with numbers).

| | Hypothesis | Observable | Predicted at 16 slots, 40 rps | Predicted at 32 slots or 20 rps |
|---|---|---|---|---|
| H1 | Adapter thrashing: loads and evictions use the time | adapter loads/evictions per s (timestamped), host-to-device bytes | loads/s and bytes high, rising with load | loads only at first touch, near zero |
| H2 | Batch fragmentation: few requests per batch because of the distinct-adapter limit | batch size, distinct adapters per batch, per step | running batch mostly 16 adapters, small mean batch | mean batch larger, more requests per adapter |
| H3 | Queue policy / starvation: waits depend on adapter popularity, and requests for loaded adapters do not wait behind unloadable ones | queue wait by adapter rank and by "adapter resident on arrival"; wait by queue position | tail-adapter requests wait long, resident-adapter requests short; wait not set by position | all waits short |
| H4 | Synchronous loads stall the scheduler | scheduler loop iteration time, split by iterations with a load | iterations with a load are long, and enough of them to add up to the lost throughput | rare |
| H5 (added) | Capacity: with 16 slots SGLang's decode throughput is below the offered load, so the queue grows regardless of policy | per-step decode time and tokens per step at 16 vs 32 slots; running-batch size | tok/s below offered 5120 and flat while queue grows | tok/s tracks offered load |

H3 and H5 can both hold. H5 is not on your list: the throughput gap in section c (about 3000 vs 4600 tok/s at 40 rps against an offered 5120) suggests it is worth checking separately from H1–H4.
Observables that need instrumentation (loads, iteration time, batch composition) may need a log-only patch in a separate worktree; check SGLang's own metrics and `--enable-metrics` first (not yet looked at).

## Open items for your decision before Phase 1
- vLLM source is the rc0 tag. I can fetch `v0.34.0` before the Phase 2 vLLM comparison; I did not want to touch the clone's remotes without asking.
- I have not read the rest of the v0.5.23 → main diff for LoRA-related changes outside the drainer (e.g. `lora_registry.py`, `lora.py`). The admission and load path checked above is unchanged.
- Whether to add `--lora-drain-wait-threshold` values other than 1.0 to Phase 1: the strict-`≤` rule suggests main behaves the same for any positive threshold, but the threshold also sets how soon a drain starts, so I would add one smaller value (0.2 s, below the study's 500 ms bar) only if you want it.

## Follow-up after your answers (Phase 0, still reading only)
- **vLLM `v0.34.0` fetched** into `~/src/vllm` (tag only). `git diff v0.34.0rc0 v0.34.0` shows no change to `vllm/v1/core/sched/scheduler.py` or `vllm/lora/`, and the `max_loras` admission check sits at the same lines (`scheduler.py:1130-1141`). Section c holds for 0.34.0.
- **Rest of the LoRA diff, v0.5.23 to main** (23 files under `srt/lora`, plus `scheduler.py` and `schedule_policy.py`): on the paths read for this question the other changes are import moves, an Intel XPU backend, pipeline-parallel `pp_loras`, per-layer buffer allocation in `mem_pool.py`, adapter-name normalization in `lora.py`, and removal of `LoRAAdapter.pin_weights_in_cpu` (defined at `lora.py:530` in v0.5.23, no caller in either tree, so dead code). The admission rule, slot choice, LRU eviction and per-load `pin_memory()` match. I did not read `schedule_policy.py`'s 1340-line diff beyond checking that it has no LoRA-specific code at either commit (no "lora" matches).
- **Phase 1 threshold set:** `--lora-drain-wait-threshold` of 1.0 and 0.2 on both v0.5.23 and main, plus default (0.0 = off) on both: 6 server configs. (v0.5.23 and main each take two thresholds plus default.)
