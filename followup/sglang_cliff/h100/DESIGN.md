# H100 check: full design table

Moved from the paper (section 6.9, brian.biz/multilora) where only the rows that differ from the RTX 4090 runs are kept. Plan: `../h100_amendment.md`. Raw data: `results/`.

| item | value |
|---|---|
| Hardware | RunPod pod `f70961d23335`: 4 x NVIDIA H100 80GB HBM3, driver 595.112.07, CUDA 13.2, container CPU quota 88.4 CPUs, Python 3.12.3. TP=1 on GPU 0, TP=2 on GPUs 0 and 1; the GPUs were checked free before each server start. Exclusive access to the physical host is not established; the pod's load average at server starts ranged from 3.2 to 27.6 (`run.log`) |
| SGLang builds | stock `a9a27e9255` (the PR's merge base with main); patched `9009897b21` (PR head: three commits plus a merge of main up to `a9a27e9255`). The diff is `memory_hook.py`, `kv_cache_configurator.py` and one CPU unit test. Separate virtual environments with the same resolved dependencies (torch 2.14.1, flashinfer 0.7.4, sglang-kernel 0.4.11; freeze files in `logs/`). Main contains PR #52709, which sizes the graph ceiling to `--max-running-requests` when that flag is set; it is not set here, so it does not apply |
| Reference engine | vLLM 0.34.0 (torch 2.14.0, flashinfer 0.7.3), as in the 4090 runs |
| Model and adapters | Qwen2.5-1.5B-Instruct; the same 32 synthetic rank-16 adapters on all 7 projection modules (B scale 0.05) as the 4090 runs, checked against the original SHA-256 manifest before any run. They are the numerically extreme adapters of section 8 of the paper |
| SGLang flags | `--enable-lora --max-lora-rank 16 --max-loras-per-batch 16 --mem-fraction-static 0.85 --context-length 4096 --tp N` with all seven projection modules as LoRA targets; the running limit and graph range left at their defaults |
| vLLM flags | `--max-model-len 4096 --gpu-memory-utilization 0.85 --enable-lora --max-loras 16 --max-cpu-loras 64 --max-lora-rank 16 --tensor-parallel-size N` |
| Open-loop workload | Poisson arrivals at 20 and 40 requests/s, 256 input and 128 requested output tokens, Zipf adapter choice over 32, 50 warm-up plus 500 measured requests, seed 1, after two 300-request warm-up passes. 3 fresh server starts per build, alternating stock and patched |
| Closed-loop workload | Concurrency 1,024, 3,000 requests after 300 warm-up, otherwise the same; 1 server start per build. Closed loop keeps 1,024 requests in flight regardless of speed, so it measures capacity, and its TTFT includes waiting behind that fixed concurrency; it is not a prediction of user latency at any offered rate |
| Load generator | `scripts/loadgen.py`, unchanged from the 4090 runs (`ignore_eos` sent to SGLang and vLLM) |
| Output metric | Same as the 4090 runs: `output_tokens_per_s` in the raw files counts non-empty streamed text chunks per second, and `tpot_ms` is time per streamed chunk. The raw files, the repository summary script and the PR #53433 comment call these tokens; here they are chunks. The fixed 128-token request length gives a check that does not depend on chunk counting: completed requests/s rose by the same ratio (102.9 to 128.4, 97.6 to 125.9) |
