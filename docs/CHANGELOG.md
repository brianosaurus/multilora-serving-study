# Changelog

Changes to the plan after the first run are recorded here with the reason. Entries before the first run are listed for completeness.

## 2026-11-22 (before any engine was tested with an adapter)

- Clock: hour 0 set to 2026-11-21 23:25:58 UTC (when the engine installs began, as the author specified); hard stop 2026-11-22 23:25:58 UTC.
- Gate 1 passed at about hour 0.6: vLLM 0.34.0 (90 s cold start), SGLang 0.5.23 (96 s), and mistral.rs v0.9.5 (25 s) each answered a base-model request. mistral.rs returned HTTP 503 on every request with its default memory settings and with `--gpu-memory-utilization 0.85` (its log reports transient-workspace memory pressure); an explicit `--kv-cache 8GiB` works and is recorded as a per-engine setting. Its plain loader does not support `--max-model-len`.
- Correctness-check criteria fixed (see the Correctness check section), approved by the author.
- Calibration run against the PEFT reference only, at about hour 0.7 (data in `results/raw/calibration/`). Adapters changing the output within 8 tokens, out of 16 prompts: scale 0.01 gave 8, 8, 5, 7 (fails); 0.02 gave 13, 12, 10, 16 (fails); 0.05 gave 16, 16, 16, 16 (passes). By the pre-registered rule, the B scale is 0.05.
- Observation, not a plan change: at scale 0.05 the adapters' generated text is incoherent (for example "The capital of France is" continues as " the largest part of a circular..."). The synthetic adapters therefore exercise serving mechanics and the routing between adapters, not the quality of a useful fine-tune, and the write-up must say so. Matrix-multiply cost does not depend on weight values, so timing is not affected. Chaotic text may make correct engines miss the 90% identity threshold through bf16 near-ties; the threshold is not moved, and any such result is reported as it is.
- Gate 2, the pre-registered criteria applied exactly as written (data in `results/raw/correctness/`). Serial and mixed accuracy: vLLM 0.34.0 71.9% and 68.8%, SGLang 0.5.23 65.6% and 65.6%, mistral.rs v0.9.5 42.2% and 35.9%. All fail the 90% threshold.
- Noise floor (`scripts/noise_floor.py`): Hugging Face with the adapter merged into the weights in bf16, a mathematically equivalent implementation, scores 43.8% under the same criterion. A known-correct implementation therefore cannot pass criteria 2 and 3 with these adapters, so the failures are not evidence about the engines. Most failures are ties on very short shared prefixes: the outputs diverge from the reference within one or two tokens. No engine's output resembles the base model's, so the adapter is applied in all three.
- Conclusion: criteria 2 and 3 as written lack power with these chaotic synthetic adapters. This is a flaw in the design, found after seeing the engine results. Any replacement is a post-hoc amendment and will be labeled as one.
- 128 adapters generated (seeds 0 to 127, scale 0.05, rank 16, float16, 37.0 MB each, 4.5 GB in RAM-backed storage; manifest with checksums in `results/raw/adapters_manifest.json`). Adapters 0 to 3 were verified byte-identical to the calibration set the reference was computed from.

## 2026-11-21 (before any run)

- Initial scaffold and plan.
- mistral.rs moved into the outcome-level comparison; fixed tuning protocol and relative metric added.
- Decisions fixed for model, adapters, adapter counts, slots, traffic, TTFT limit, metrics, repeats, replication thresholds, versions, and GPU hygiene. Feasibility checks not yet done.
- Day plan added because the author requires the whole study to run in under one day: reduced grid (adapter counts 1 and 32, fixed slots, a rate ladder, 3 repeats, 500 requests per run), three timed gates, a cut order, and a hard stop at hour 24. The tuning settings list is still open and must be committed before the first timed run.
- Storage row updated to the machine's existing network-mounted volume (about 47 GiB usable, measured). The machine was wiped of old data and is dedicated to this study. Write ability and speeds are confirmed; the install-time and load-leak checks remain.
- Storage row added (50 GB network volume for installs and results; weights and adapters copied to RAM-backed storage before timed runs; caches local) with three feasibility checks.
- Relative metric defined at two levels (adapter versus no adapter within each engine, then across engines). Single-GPU constraints and a reference-run drift check added. Only the tuning protocol remains open.
