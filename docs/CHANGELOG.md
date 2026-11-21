# Changelog

Changes to the plan after the first run are recorded here with the reason. Entries before the first run are listed for completeness.

## 2026-11-21 (before any run)

- Initial scaffold and plan.
- mistral.rs moved into the outcome-level comparison; fixed tuning protocol and relative metric added.
- Decisions fixed for model, adapters, adapter counts, slots, traffic, TTFT limit, metrics, repeats, replication thresholds, versions, and GPU hygiene. Feasibility checks not yet done.
- Day plan added because the author requires the whole study to run in under one day: reduced grid (adapter counts 1 and 32, fixed slots, a rate ladder, 3 repeats, 500 requests per run), three timed gates, a cut order, and a hard stop at hour 24. The tuning settings list is still open and must be committed before the first timed run.
- Storage row updated to the machine's existing network-mounted volume (about 47 GiB usable, measured). The machine was wiped of old data and is dedicated to this study. Write ability and speeds are confirmed; the install-time and load-leak checks remain.
- Storage row added (50 GB network volume for installs and results; weights and adapters copied to RAM-backed storage before timed runs; caches local) with three feasibility checks.
- Relative metric defined at two levels (adapter versus no adapter within each engine, then across engines). Single-GPU constraints and a reference-run drift check added. Only the tuning protocol remains open.
