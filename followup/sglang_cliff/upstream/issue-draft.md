# DRAFT (not posted)

**Title:** [Bug][Perf] Decode batches above the largest captured CUDA graph run eagerly and the server stays in a slow state (multi-LoRA reproduction of #33483, still on main)

## Summary

On an RTX 4090 the decode CUDA graph range defaults to 48 (`memory_hook.py`, tp < 4), while `max_running_requests` is derived from KV capacity (4096 here). Nothing keeps the running batch within the graph range. Once a decode batch passes 48 it runs eagerly; on a launch-bound setup an eager step costs about 15 times a replay, requests finish more slowly, more of them accumulate, and the batch stays above 48.

**Once eager decoding starts, the server does not recover while the load continues.** At 40 rps on default settings, about a fifth of requests (110 of 500) wait more than 1 s for their first token, and every one of them (110 of 110, in both starts) gets it only after the load generator has sent its last request. They are served when arrivals stop, not before. The backlog takes 9.5 s to drain after the last arrival (1.8 s with the fix below). In a 20 rps start that tipped into the slow state two thirds of the way through, the p90 TTFT went from 0.1 s to 15 s and stayed above 7 s until the run ended. With the fix, the same 40 rps load gives a p95 TTFT of 2.1 to 2.3 s, against 13.4 to 13.5 s.

This is the mechanism reported in #33483 (L40, 0.5B model), which was closed for inactivity without a fix. It still reproduces on main, and two things make it worse than that report suggests:

- **LoRA.** With adapters enabled, an eager decode step costs about 31 ms against 1.7 to 2.0 ms for a replay (no LoRA: 18.7 ms against 1.3 ms).
- **CPU contention alone triggers it.** With 14 busy-loop processes in the same container (17.85-CPU quota, not throttled), a server that handles 20 rps at 82 ms p95 TTFT went to 10.6 s in one of two starts. It also happened in one of several quiet starts.

## Setup

- RTX 4090 24 GB, 1 GPU, SGLang `0e9cff98f5` with main `385262cabb`'s python tree for the fix; on main the 24 GB tier's range (48) and the KV-derived running limit are unchanged.
- Qwen2.5-1.5B-Instruct, 32 synthetic rank-16 adapters on all 7 projection modules, `--max-loras-per-batch 16`, csgmv backend, `--mem-fraction-static 0.85 --context-length 4096`, otherwise defaults.
- Open-loop Poisson load, 256 input / 128 output tokens, adapters chosen by Zipf over 32, 500 requests after 50 warm-up, seed 1.
- Shared cloud host: timings are screening numbers, repeated across starts.

```bash
python -m sglang.launch_server --model-path Qwen/Qwen2.5-1.5B-Instruct --enable-lora \
  --max-lora-rank 16 --max-loras-per-batch 16 \
  --lora-target-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --lora-paths adapter_0000=... adapter_0031=... --mem-fraction-static 0.85 --context-length 4096
```

Harness, raw traces and scripts: <study repo link>.

## Evidence

Median wall time per decode step by batch size (closed loop):

| server | batch 1 to 48 | batch 49 to 96 |
|---|---|---|
| LoRA, default range (48) | 1.7 to 1.9 ms | 31.4 ms (eager) |
| LoRA, `--cuda-graph-max-bs-decode 256` | 1.7 to 1.9 ms | 2.0 ms |
| no LoRA, default | 1.3 ms | 18.7 ms (eager) |
| no LoRA, 256 | 1.2 ms | 1.3 ms |

The eager cost is flat across batch 49 to 96, so it is per-step CPU work (Python and launches), not GPU work. That is also why CPU contention makes it worse: the scheduler thread is CPU-bound (about 85% busy at 20 rps) and its steps get longer.

40 rps, 2 starts each (p95 TTFT; median TPOT; output tok/s). "Busy loops": 14 `while True: pass` processes in the same container.

| config | no contention | 14 busy loops |
|---|---|---|
| default | 13.5 s, 13.4 s; 43 to 44 ms; 2981, 3040 | 14.5 s, 18.9 s; 53 to 76 ms; 2725, 2155 |
| **fix (PR)** | **2.25 s, 2.08 s; 12.2 ms; 4665, 4673** | **3.35 s, 7.93 s; 14 to 16 ms; 4598, 4014** |
| vLLM 0.34.0, same model, adapters, load | 1.43 s, 1.43 s; 10 ms; 4657, 4658 | 1.44 s, 1.46 s; 10 ms; 4671, 4446 |
| `--cuda-graph-max-bs-decode 256` only (earlier session) | 2.2 s, 6.2 s; 13 to 16 ms; 4664, 4265 | 7.7 s, 9.2 s; 4407, 3913 |
| `--max-running-requests 48` only (earlier session) | 1.8 s, 1.9 s; 9.4 ms; 4524, 4471 | 7.9 s, 13.3 s; 3162, 2384 |

GSM8K (sgl-eval 0.1.3, greedy, 128 concurrent): default 67.5%; fix 66.3%; range 256 alone 66.3% and 66.5%.

## Root cause

`max_running_requests` and the decode graph range are set by unrelated code (KV capacity, and a GPU-memory table), and no path reconciles them. vLLM bounds both together: its running limit (`max_num_seqs`, 256 here) sets the capture range (to 512 here), so decode never leaves the graph.

## Proposed fix (PR <link>)

Do the same in SGLang:
1. When `--max-running-requests` is not set, default it to the largest decode graph batch size (logged, with a hint to raise `--cuda-graph-max-bs-decode`). An explicit value is unchanged.
2. Raise the 24 GB tier's default range for tp < 4 from 48 to 256, like #37898 and #46132 for larger tiers. Measured on the 4090: graphs to 256 take 0.54 GB (0.15 GB to 48); the KV pool is unchanged at `--mem-fraction-static 0.85` (600,201 tokens both ways).

Alternatives considered: only raising the range (the batch can still pass 256, since the running limit stays 4096); only capping at 48 (limits concurrency); splitting an oversized batch into several graph replays inside one step (prototyped; it races with the overlap scheduler and attention planning on shared buffers, and correct versions still cost 11 to 20 ms per step).

The remaining gap to vLLM at 40 rps (2.1 to 2.3 s against 1.4 s) and under contention is not addressed here; traces point at adapter-load stalls, for a separate issue.

## Limits

One model (1.5B), one GPU, shared host. Not measured: larger models, where a range of 256 may need memory the KV pool would otherwise use, and other 24 GB GPUs.
