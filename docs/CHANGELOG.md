# Changelog

Changes to the plan after the first run are recorded here with the reason. Entries before the first run are listed for completeness.

## 2026-11-22 (before any engine was tested with an adapter)

- Clock: hour 0 set to 2026-11-21 23:25:58 UTC (when the engine installs began, as the author specified); hard stop 2026-11-22 23:25:58 UTC.
- Gate 1 passed at about hour 0.6: vLLM 0.34.0 (90 s cold start), SGLang 0.5.23 (96 s), and mistral.rs v0.9.5 (25 s) each answered a base-model request. mistral.rs returned HTTP 503 on every request with its default memory settings and with `--gpu-memory-utilization 0.85` (its log reports transient-workspace memory pressure); an explicit `--kv-cache 8GiB` works and is recorded as a per-engine setting. Its plain loader does not support `--max-model-len`.
- Correctness-check criteria fixed (see the Correctness check section), approved by the author.

## 2026-11-21 (before any run)

- Initial scaffold and plan.
- mistral.rs moved into the outcome-level comparison; fixed tuning protocol and relative metric added.
- Decisions fixed for model, adapters, adapter counts, slots, traffic, TTFT limit, metrics, repeats, replication thresholds, versions, and GPU hygiene. Feasibility checks not yet done.
- Day plan added because the author requires the whole study to run in under one day: reduced grid (adapter counts 1 and 32, fixed slots, a rate ladder, 3 repeats, 500 requests per run), three timed gates, a cut order, and a hard stop at hour 24. The tuning settings list is still open and must be committed before the first timed run.
- Storage row updated to the machine's existing network-mounted volume (about 47 GiB usable, measured). The machine was wiped of old data and is dedicated to this study. Write ability and speeds are confirmed; the install-time and load-leak checks remain.
- Storage row added (50 GB network volume for installs and results; weights and adapters copied to RAM-backed storage before timed runs; caches local) with three feasibility checks.
- Relative metric defined at two levels (adapter versus no adapter within each engine, then across engines). Single-GPU constraints and a reference-run drift check added. Only the tuning protocol remains open.
