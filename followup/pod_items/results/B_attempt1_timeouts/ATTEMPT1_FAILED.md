# Item B, attempt 1: failed by timeout, NOT a result

Pod clock 2026-11-24 08:02 to about 08:47 UTC, host `runpod` (126ab78ffbd7). The first four starts (vLLM 16, 32, 64 slots; SGLang 16 slots) hit the 10-minute readiness limit while Python start-up took 3 to 10 minutes on the shared host (load 25 to 100). Rows with `ready=0` and any `gpu_mem` value are timeouts, not measurements of capacity or memory. Nothing here is used in any table.
The rerun with a 30-minute limit is in `../B/` (attempt 2).
