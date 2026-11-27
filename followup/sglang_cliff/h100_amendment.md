# H100 check for SGLang PR #53433 (pre-registered; written before any H100 data)

Date: 2026-11-27. PR #53433 is open, not merged. AI assistance: plan, scripts and analysis drafted with Claude Code; the author reviews and approves each step.

## Question

Does the PR regress SGLang on an H100? On this GPU tier the PR's range raise (24 GB tier, 48 to 256) does not apply: the H100 tier already captures decode graphs to 256 (tp < 4) or 512 (tp = 4). What does apply is the PR's other change: when `--max-running-requests` is not set, the running limit defaults to the largest decode graph batch size instead of the KV-derived value. On an 80 GB GPU with a 1.5B model the KV-derived value is far above 256, so this check is mainly about whether that cap costs throughput or latency.

## Hardware and software

- RunPod pod `f70961d23335`: 4 x H100 80GB HBM3, driver 595.112.07, CUDA 13.2, container CPU quota 88.4 CPUs (22 per GPU), Python 3.12.3. Recorded in `h100/env.json`.
- Stock SGLang: `a9a27e9255`, the merge base of the PR with main.
- Patched SGLang: `9009897b21`, the PR head (the PR's three commits plus a merge of main up to `a9a27e9255`). Diff against stock: `memory_hook.py`, `kv_cache_configurator.py` and one unit test, nothing else.
- Both builds are installed from source into separate virtual environments (`/workspace/envs/sglang-stock`, `/workspace/envs/sglang-patched`) with the same resolved dependencies; the resolved versions are recorded at setup.
- vLLM 0.34.0 in `/workspace/envs/vllm`, the version used in the 4090 runs.
- Context: main now contains #52709, which raises the decode graph range to cover `--max-running-requests` when that flag is set. With the flag unset (the case tested here) it does not apply, so stock still derives the running limit from KV capacity.

## Workload (copied from the 4090 cliff runs, step 10 part AA; no new workload)

- Model: Qwen2.5-1.5B-Instruct. Adapters: 32 synthetic rank-16 adapters on all 7 projection modules, seeds 0 to 31, B scale 0.05, made with `scripts/make_adapters.py --count 32 --b-std 0.05`; their SHA-256 must equal `results/raw/adapters_manifest.json` or no run starts.
- SGLang flags: `--enable-lora --max-lora-rank 16 --max-loras-per-batch 16 --lora-target-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj --mem-fraction-static 0.85 --context-length 4096`, plus `--tp N`. Nothing else set; in particular `--max-running-requests` and `--cuda-graph-max-bs-decode` are left at their defaults.
- vLLM flags: `--max-model-len 4096 --gpu-memory-utilization 0.85 --enable-lora --max-loras 16 --max-cpu-loras 64 --max-lora-rank 16`, plus `--tensor-parallel-size N`.
- Harness: `scripts/loadgen.py`, copied unchanged. Open-loop Poisson, 256 input and 128 output tokens, Zipf adapter choice over 32, 50 warm-up plus 500 measured requests, seed 1, after the same two 300-request warm-up passes the 4090 runs used.

## Runs (in this order; one server and one GPU workload at a time; GPUs checked free between runs)

1. tp = 1, stock and patched SGLang: 20 rps and 40 rps, 3 fresh server starts per build, starts alternating stock, patched. Then one saturating run per build: closed loop at concurrency 1024 (`loadgen.py --concurrency 1024 --requests 3000 --warmup 300`, otherwise the same workload).
2. tp = 1, vLLM: 20 rps and 40 rps, 3 starts (reference).
3. GSM8K, tp = 1, stock and patched: sgl-eval 0.1.3 in its own environment, full set (1319 problems), greedy (`--temperature 0 --max-tokens 512 --seed 0`), `--num-threads 128`, base model on the LoRA server, as in step 9 part Z1. Standard error: sqrt(p(1 - p) / 1319).
4. Stop and report if the patched build's saturating throughput is more than 5% below stock's, or if anything crashes or runs out of memory. Otherwise: steps 1 and 2 at tp = 2, then tp = 4, and GSM8K for the patched build at tp = 4.

At every server start, log: `max_running_requests`, the decode CUDA graph batch sizes, KV pool tokens and CUDA graph memory. For the whole run, log CPU (`mpstat 1`) and GPU (`nvidia-smi dmon`) utilization. Budget: if wall time passes 5 hours, finish the current run, stop and report.

## Metrics

p50 and p95 TTFT, median TPOT and output tokens per second for each start; saturating output tokens per second; the start-time values above; mean CPU utilization; GSM8K accuracy with its standard error.

## Decision rule

The PR "does not regress" on H100 at a given tp if both hold:

- Saturating throughput: patched is at least 0.95 times stock.
- Latency: at 20 rps and at 40 rps, the median over 3 starts of patched p95 TTFT is at most 1.10 times stock's. (The 10% allowance is for start-to-start noise on a shared host; I set it here, before any data, and the author can tighten it.)

GSM8K is reported with its standard error; a gap larger than 2 standard errors between stock and patched is flagged.

## Predictions

- (h1) Stock tp = 1 starts with a running limit far above 256; patched starts with 256 and a graph range of 256 in both builds.
- (h2) At 20 and 40 rps neither build reaches 256 running requests, so p95 TTFT is the same within noise.
- (h3) At concurrency 1024 the patched build queues requests above 256 while stock runs them, some eagerly above 256. Which is faster is not predicted.
- (h4) GSM8K within 2 standard errors.
