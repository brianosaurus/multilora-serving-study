# DRAFT (not posted) - ON HOLD until the split PR passes accuracy

**Title:** [Bug][Perf] Decode batches above the largest captured CUDA graph run eagerly and the server stays in a slow state (multi-LoRA reproduction of #33483, still on main)

## Summary

On an RTX 4090 the decode CUDA graph range defaults to 48 (`memory_hook.py`, tp < 4), while `max_running_requests` is derived from KV capacity (4096 here). Nothing keeps the running batch within the graph range. Once a decode batch passes 48 it runs eagerly; on a launch-bound setup an eager step costs about 15 times a replay, requests finish more slowly, more of them accumulate, and the batch stays above 48. This is the mechanism reported in #33483 (L40, 0.5B model), which was closed for inactivity without a fix. It still reproduces on main, and two things make it worse than that report suggests:

- **LoRA.** With adapters enabled, an eager decode step costs about 31 ms against 1.7 to 2.0 ms for a replay (no LoRA: 18.7 ms against 1.3 ms).
- **CPU contention alone triggers it.** With 14 busy-loop processes in the same container (17.85-CPU quota, not throttled), a server that handles 20 rps at 82 ms p95 TTFT went to 10.6 s in one of two starts, and in earlier runs 2 of 3 starts went slow. Without contention it also happened in one of several quiet starts.

## Setup

- RTX 4090 24 GB, 1 GPU, SGLang `0e9cff98f5`. On main `385262cabb` the 24 GB tier's graph range (48) and the KV-derived running limit are unchanged (checked in the code, not re-measured).
- Qwen2.5-1.5B-Instruct, 32 synthetic rank-16 adapters on all 7 projection modules, `--max-loras-per-batch 16`, csgmv backend, `--mem-fraction-static 0.85 --context-length 4096`, otherwise defaults.
- Open-loop Poisson load, 256 input / 128 output tokens, adapters chosen by Zipf over 32, 500 requests after 50 warm-up, seed 1.
- Shared cloud host: timings are screening numbers, repeated across starts, not dedicated-machine measurements.

```bash
python -m sglang.launch_server --model-path Qwen/Qwen2.5-1.5B-Instruct --enable-lora \
  --max-lora-rank 16 --max-loras-per-batch 16 \
  --lora-target-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --lora-paths adapter_0000=... adapter_0031=... --mem-fraction-static 0.85 --context-length 4096
```

Harness, raw traces and the scripts behind every number: <study repo link>.

## Evidence

Median wall time per decode step by batch size (closed loop, no contention):

| server | batch 1 to 48 | batch 49 to 96 |
|---|---|---|
| LoRA, default graph range (48) | 1.7 to 1.9 ms | 31.4 ms (eager) |
| LoRA, `--cuda-graph-max-bs-decode 256` | 1.7 to 1.9 ms | 2.0 ms |
| no LoRA, default | 1.3 ms | 18.7 ms (eager) |
| no LoRA, 256 | 1.2 ms | 1.3 ms |

The eager cost is flat across batch 49 to 96, so it is per-step CPU work (Python and launches), not GPU work. That is also why CPU contention makes it worse.

40 rps, no contention, 2 starts each (p95 TTFT, median TPOT, output tok/s):

| config | p95 TTFT | TPOT | tok/s |
|---|---|---|---|
| default | 13.5 s, 13.5 s | 43 to 44 ms | 2986, 3014 |
| `--cuda-graph-max-bs-decode 256` | 2.2 s, 6.2 s | 13 to 16 ms | 4664, 4265 |
| `--max-running-requests 48` | 1.8 s, 1.9 s | 9.4 ms | 4524, 4471 |
| vLLM 0.34.0, same model, adapters and load | 1.5 s, 1.5 s | 10.2 ms | 4646, 4655 |

In slow starts, the share of decode steps above 48 tracks how slow the start is (10 to 65% of steps, against 0% in normal starts, where batches never pass 36). Scheduler-loop timing shows the scheduler thread is CPU-bound (about 85% busy at 20 rps) and is not preempted under contention; its steps get longer.

## Root cause

`max_running_requests` and the decode graph range are set by unrelated code (KV capacity, and a GPU-memory table), and no path reconciles them. vLLM avoids this by deriving its capture range from `max_num_seqs` (on a 4090: 256 running, graphs to 512).

## Options

| option | cost |
|---|---|
| raise the 24 GB tier's default range (like #46132 for 35 to 60 GiB) | KV memory, about 2 MB per unit of range; small for 1.5B, about 12% of the KV pool for an 8B bf16 model on 24 GB |
| default `max_running_requests` to the graph range | no memory; limits concurrency, which may cost throughput where a large eager batch would be faster |
| run an oversized decode batch as several graph replays | no memory, no concurrency limit; more code. PR: <link> |

I have a PR for the third option with measurements. Happy to adjust direction if maintainers prefer another.

## Limits

One model (1.5B), one GPU, 16 adapter slots, shared host. Larger models, where eager decode costs less relative to a replay, are not measured.
