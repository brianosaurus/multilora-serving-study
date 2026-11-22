# Differential study: changelog

Rule: PLAN.md is committed before any measured run. After that, every change to the plan, thresholds, versions, harness or cells is recorded here with the date and the reason. History is never rewritten.

| Date | Change | Reason |
|---|---|---|
| 2026-11-22 | Initial PLAN.md drafted (Gate 1). | AI-drafted (Claude); not yet reviewed by Brian. No measured run has happened. |
| 2026-11-22 | Reference definition clarified and regenerated: `repetition_penalty=1.0` explicit (raw greedy). Added cell C8 (request with only `temperature=0`, compared with the HF-default reference). Axis-1 phase 1 for Qwen rerun; first-pass results kept as superseded. | **Found after phase 1 ran** (so this is a post-run change): the first Qwen reference was produced with the HF default generation config, which applies Qwen2.5-1.5B's `repetition_penalty=1.1` even with `do_sample=False`; 537 of 6,067 reference tokens (8.9%) were not the raw top-1 token. The plan specified plain greedy. Teacher-forced scores were largely unaffected, but every generation-based comparison and the mistral.rs fallback were. Detected by checking chosen-token rank against raw top-20; fixed before any result was interpreted. |
