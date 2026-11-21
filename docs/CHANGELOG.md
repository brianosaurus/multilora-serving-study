# Changelog

Changes to the plan after the first run are recorded here with the reason. Entries before the first run are listed for completeness.

## 2026-11-21 (before any run)

- Initial scaffold and plan.
- mistral.rs moved into the outcome-level comparison; fixed tuning protocol and relative metric added.
- Decisions fixed for model, adapters, adapter counts, slots, traffic, TTFT limit, metrics, repeats, replication thresholds, versions, and GPU hygiene. Feasibility checks not yet done.
- Relative metric defined at two levels (adapter versus no adapter within each engine, then across engines). Single-GPU constraints and a reference-run drift check added. Only the tuning protocol remains open.
