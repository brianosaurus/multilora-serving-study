# Dedicated session: runtime estimate and machine requirements (2026-11-24; estimates, not measurements)

Assumptions (stated so they can be checked): a measured cell is 550 open-loop requests (50 warm-up, 500 measured) at 10 to 80 rps, about 55 s at 10 rps and 7 s at 80 rps to send, plus drain time when the server is past its cliff (up to about 60 s), plus the 6 s external-CPU check before it; I use 75 s per cell on average. A vLLM or SGLang server start with LoRA takes about 2.5 to 3 minutes for the 1.5B model (measured on the pod: vLLM 1.5 min after caches, SGLang 2 to 3 min), plus about 1 minute for the two warm-up runs and two reference runs. A Llama-3.1-8B start is taken as 4 minutes. A quiet machine has no wait for external CPU; every wait up to 120 s per cell on a noisy machine adds up to 2 minutes a cell (not included).

## Blocks of the existing plan (B1 to B5)

| Block | What | Estimate |
|---|---|---|
| B1 | 3a repeat, SGLang cold and warm passes | 10 min |
| B2 | 4a kernel rerun: 6 blocks x 3 binaries x (server start about 8.5 min with flash-attn + 8 cells x 150 requests) | about 150 min (the earlier attempt's own estimate) |
| B3 | 4b: 4 server sessions x (4 min start and warm-up + 18 cells x 75 s = 22 min) | about 105 min |
| B4 | 3b: 4 sessions at 16 slots x (4 + 15 cells x 75 s = 19 min) + 2 sessions at 32 slots x (4 + 3 cells x 75 s = 4 min) | about 115 min |
| B5 | 3c: 2 sessions x (6 min start with 128 adapters + 6 cells x 75 s = 8 min) | about 30 min |
| Total | | **about 410 min = 6.8 hours** (range 6.5 to 8 hours) |

## csgmv A/B (`followup/csgmv_ab/PLAN.md`, S0 to S4; not yet in `run_session.sh`)

Each step runs both backends, 3 repeats, a server restart per backend per repeat (6 starts).

| Step | Load | Estimate |
|---|---|---|
| Setup | download Llama-3.1-8B-Instruct (about 16 GB; a gated model, needs a Hugging Face token with the licence accepted) and the adapter `algoprog/fact-generation-llama-3.1-8b-instruct-lora`, the ShareGPT file (about 0.7 GB) | 5 to 15 min, depends on the link |
| Fit check | start the 8B server with `--mem-fraction-static 0.85` and the LoRA flags, read the KV capacity from the log (8B bf16 weights are about 16 GB of 24 GB, so the KV cache is small, a few GB; this decides whether `--mem-fraction-static` or context length must change; it is recorded in the plan before S0) | 10 min |
| S0 | 8B, 1 adapter, radix off, `bench_serving` 1000 prompts at 2, 8, 16, inf rps: about 500 + 125 + 80 + 150 s = 14 min per backend-repeat x 6 = 85 min, + 6 starts x 4 min = 24 min | about 110 min |
| S1 | Qwen2.5-1.5B instead: rates 2 and 8 take the same 625 s, 16 and inf about 100 s: 12 min x 6 = 72 min + 6 starts x 2.5 min | about 87 min |
| S2 | S1 with 32 Zipf adapters on 16 slots | about 87 min |
| S3 | S2 with the radix cache on | about 87 min |
| S4 | S3, closed loop, concurrency 32 and 128, 500 requests, 3 repeats: 12 runs x 30 s + 6 starts | about 25 min |
| Total A/B | | **about 410 min = 6.8 hours** (about 7 to 8 with the download) |

The rate 2 runs (500 s each, 6 per step) are the largest single cost: 50 of the 110 minutes in S0. If the author wants it shorter, a lower request count at rates 2 and 8 would be a plan change to decide before the run (it changes what is compared with #10286's 1000-prompt numbers).

## Totals

- Without the A/B: **about 6.8 hours** (6.5 to 8).
- With the A/B: **about 14 hours** (13.5 to 16), i.e. two sessions of 7 hours, or one session of 14 hours with the quiet checks throughout. The A/B is not in `run_session.sh` yet; adding it is a script change to make (and dry-run) before that estimate applies.

## Machine requirements

- **GPU:** one NVIDIA RTX 4090 (24 GB, sm_89), the same as the study; nothing else using it; persistence mode on; no power cap below stock. A different GPU makes every number incomparable with the study's.
- **CPU:** dedicated vCPUs, not a shared-host container (the pod reported 64 cores but other tenants used 6 to 17 of them, which is the reason for this move). At least 16 dedicated vCPUs (the load generator, the server's CPU threads and the quiet check all run together; builds would want 32), and a way to verify nobody else uses the host (the script's check: external CPU at most 1 core).
- **RAM:** at least 64 GB, of which `/dev/shm` at least 40 GB (the model 3 GB, 128 adapters 4.7 GB, 32 more adapters 1.2 GB, the 8B model 16 GB if the A/B runs). The pod had 440 GB.
- **Disk:** local NVMe (not a network volume), at least 150 GB free: the three Python environments 31 GB (the `sglang_main` copy is 11 GB if it cannot share files with the v0.5.23 one), Qwen 3 GB, Llama-3.1-8B 16 GB, ShareGPT 0.7 GB, mistral.rs binaries 1.1 GB, results about 1 GB, and room to rebuild a mistral.rs tree (10 GB or more for a build). The pod has 50 GB total.
- **Driver and CUDA:** NVIDIA driver supporting CUDA 13 (R580 or later; the pod has 595.112.07, because the vLLM environment's torch is `2.14.0+cu130`); CUDA toolkit 12.8 (`nvcc` 12.8, as in `followup/kernel_rerun/BUILD.md`) if any mistral.rs binary is built there; the binaries copied from the pod need only the driver.
- **Software:** Linux, root, Python 3.12, `uv`, `git`, `curl`, Rust stable with cargo only if building (SGLang `main` also needs cargo to install); outbound internet during setup (PyPI, GitHub, Hugging Face).
- **Network and access:** SSH; a fixed hostname (the script refuses any other); `nvidia-smi` showing no compute processes.
- Everything is checked by `followup/dedicated_session/preflight.sh` on the machine before the session; the session itself refuses to start unless the quiet checks pass.
