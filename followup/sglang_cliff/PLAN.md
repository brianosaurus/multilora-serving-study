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
