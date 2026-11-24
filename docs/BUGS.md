# Bug log

Observations made while running the study that look like fixable defects in an engine. This is exploratory and separate from the pre-registered measurements. Each entry says what has been verified and what has not. Nothing here has been reported upstream yet.

Machine: one RTX 4090 (24 GB). Engines: vLLM 0.34.0, SGLang 0.5.23, mistral.rs v0.9.5. Model: Qwen2.5-1.5B-Instruct.

## B1. SGLang: the JIT kernel build fails unless the environment's bin directory is on PATH

- Symptom: starting the server with the virtual environment's Python by absolute path (`/path/to/venv/bin/python -m sglang.launch_server ...`, as systemd units and containers often do) fails with `FileNotFoundError: [Errno 2] No such file or directory: 'ninja'`, and the scheduler dies.
- Verified: reproduced on SGLang 0.5.23 in a fresh venv; the same command succeeds once the venv's `bin` is on PATH. `ninja` is installed in the venv.
- Cause (read in source on `upstream/main`): `python/sglang/kernels/jit/utils/compile/ninja.py` line 156 builds `["ninja", "-f", ...]` and runs it with `subprocess.run`, so ninja is found only through PATH. The neighboring `toolchain.py` resolves `nvcc` explicitly (`CUDA_HOME`, then `shutil.which`).
- Candidate fix: resolve ninja the same way, with `shutil.which("ninja")` and then the directory of `sys.executable` as a fallback, plus a unit test with PATH cleared. Small and low risk.
- Not yet verified: that `upstream/main` still fails at runtime (only the code was read); whether an issue or PR already exists (a search found none, but the search was shallow).

## B2. mistral.rs: every request returns HTTP 503 with the default memory settings on a 24 GB GPU

- Symptom: `mistralrs serve -m Qwen2.5-1.5B-Instruct -p 1234` starts, reports ready, then every completions and chat request returns `{"error":{"message":"The service is temporarily unavailable.","type":"server_error","code":"service_unavailable"}}`. The server log says `Memory pressure prevented forward admission requiring 641272832 bytes ... of transient workspace`. The same happens with `--gpu-memory-utilization 0.85`.
- Verified: reproduced twice on v0.9.5. `--kv-cache 8GiB` fixes it (valid replies, 12.2 GB used). About 4 GB of the GPU is physically free at the time.
- Evidence from the log: engine limit 20470 MiB, existing occupancy 3837 MiB, KV budget 16021 MiB, runtime reservation 612 MiB, then "Allocating 16022 MiB". Headroom inside the limit is 20470 - 3837 - 16022 = 611 MiB, while the workspace needs 641272832 bytes = 611.57 MiB.
- Hypothesis (not verified): the KV cache is sized in whole blocks and ends up 1 MiB above its budget, which consumes the 1 MiB of slack in an exactly-fitted runtime reservation, so admission (checked against the engine limit) fails while allocation (checked against physical memory) succeeded. The code path is `engine/step_memory.rs` (the rejection) and `paged_attention/cache_allocation.rs` (sizing, with a step meant to preserve the reservation).
- To verify: run at `--gpu-memory-utilization 0.84` and `0.80` (the headroom theory predicts both work), and rerun on current `master` (the relevant files exist there; the tag `v0.9.5` is not in the local clone).
- Candidate fix: size the KV cache so the reservation is preserved inside the engine limit with a margin, or make admission and allocation use the same accounting.

## B3. mistral.rs: `/v1/completions` log-probabilities use a non-standard layout

- Symptom: a completions request with `logprobs: 5` returns `{"content":[{"token":" Paris","logprob":-1.2086,"bytes":[...],"top_logprobs":[{"token":12095,"logprob":-1.2086,"bytes":" Paris"},...]}]}`. In `top_logprobs`, `token` holds the numeric token id and `bytes` holds the decoded text. In the OpenAI format, `token` is the text and `bytes` is a list of integers; the legacy completions format is different again (`tokens`, `token_logprobs`, `top_logprobs` as a list of text-to-logprob maps).
- Verified: observed in a raw response on v0.9.5. Clients written for the OpenAI formats do not parse it (this study's own harness failed on it first).
- Verified on `master` (21da642, read in source): `mistralrs-core/src/sampler.rs:549` defines `TopLogprob { token: u32, logprob: f32, bytes: Option<String> }`, while `response.rs` defines the outer `ResponseLogprob { token: String, logprob, bytes: Option<Vec<u8>>, top_logprobs }`. Within one response, `token` is a string in one place and a numeric id in the other, and `bytes` is a list of integers in one and a string in the other. So the layout is internally inconsistent, not only different from OpenAI.
- Not verified (rest of this list): whether `master` differs; whether it is deliberate; whether an issue exists. The request types are in `mistralrs-server-core/src/openai.rs` and the handler in `completions.rs`.
- Candidate fix: match the OpenAI field meanings (and the legacy completions layout for that endpoint), with a test.

## B4. mistral.rs: `--max-model-len` is listed by `serve --help` but rejected by the plain safetensors loader

- Symptom: `Error: max_model_len=4096 is not supported by this model loader` for a plain safetensors model.
- Verified on v0.9.5. May be intended; low priority. A clearer message or documentation of which loaders support it would help.

## B5. SGLang packaging: 0.5.23 needs a pre-release dependency

- `sglang==0.5.23` depends on `flash-attn-4>=4.0.0b18`, a pre-release. `uv pip install sglang==0.5.23` fails to resolve until that package is named explicitly. Plain `pip` may behave differently, so this may be a tooling difference and not a defect. Low priority.

## Planned exploratory probes

These run only when no measurement is in progress, because the GPU must be used by one process at a time. They are exploratory and are reported separately from the measured results: adapter swap storms with more adapters than slots (memory growth, errors, crashes); concurrent mixed-adapter load for cross-adapter contamination; error handling for an unknown adapter name, a wrong rank, and malformed requests (expect a client error, not a crash or a 500); long-run stability (memory growth, error rate over time); and repeated identical requests for determinism.

## Leads checked and closed

- mistral.rs v0.9.5 batching of mixed adapters: not a per-adapter loop. `lora/dynamic/cuda.rs` groups rows by adapter slot and launches the CUDA delta kernel over the active slots, with a test for mixed slots. Not a finding.
- mistral.rs adapter dtype: the fast CUDA path falls back silently when adapter and activation dtypes differ, but `loader.rs` sets the adapter dtype to the activation dtype, so our float16 adapters on a bf16 model do not trigger it. Not a finding.

## Big-finding hypotheses (to be tested by the measured runs; none is a result yet)

Each is stated so the data can contradict it.

1. Adapter overhead differs by engine by a large factor (relative metric, level 2). Test: overhead factors from B0, B1, one-adapter and 32-adapter runs.
2. Having LoRA enabled costs throughput even when no adapter is used (B1 vs B0), which would mean the cost is paid by everyone on a LoRA-capable deployment. Test: B1 / B0.
3. Skewed (Zipf) traffic is cheaper than uniform for engines that cache hot adapters, and an engine with fewer slots than adapters degrades sharply under uniform traffic. Test: zipf32 vs uni32, with 16 slots against 32 adapters.
4. Whether models that adapt k and v projections lose prefix-cache sharing across adapters (SGLang issue #49624). Test: shared-prefix workload with different adapters.
5. What other model families need (MoE experts, GQA ratios, larger ranks, quantized bases) is outside this one-model study, and will be listed as untested, not guessed.

## B6. mistral.rs: erratic latency under load in the tuning run (candidate, unexplained)

- Observed in the tuning data (`results/raw/tuning/`, Qwen2.5-1.5B, base model, no adapters): with default settings, p95 TTFT was 10 to 27 s at 10, 20 and 40 requests per second, with about 850 to 985 output tokens per second. With `--max-seqs 256` the same engine had p95 TTFT 10.2 s at rate 10 (the first rate after start), 104 ms at rate 20, and 4.2 s at rate 40. A load level with low latency sitting between two high-latency ones is not what a queueing limit produces.
- Decode time per token was 28 to 34 ms at low load, against 6.7 ms for vLLM in the same test.
- Not verified: whether this is a first-run effect (rate 10 always ran first after start), a scheduling limit, or noise. Each point is one run. To check: repeat with a warm-up pass and reversed rate order; read the server log for scheduler settings.
- Consequence for the study: mistral.rs's tuning winner (T2) rests on one passing run, so its main results need a warm-up pass and must be read with that in mind.

## Upstream search results (2026-11-22, pod clock about 04:45 UTC; GitHub issue and PR search, read-only; searches were short keyword queries and could miss things)

- B1 SGLang ninja lookup: already addressed by two open PRs, #47284 ("Use the ninja package's binary when ninja is not on PATH", opened 2026-10-29) and #50890 ("Resolve the ninja executable beyond $PATH in the JIT builder", opened 2026-11-17). Not a contribution target.
- B2 mistral.rs 503: already reported as mistral.rs #2502 (closed 2026-10-21; same log line, on an 8 GB GPU) and fixed by merged PR #2503 ("fix runtime activation overestimation"), after v0.9.5 (commit dated 2026-10-19). The root cause is therefore not the rounding hypothesis written above. Not a contribution target. Not yet rechecked on master on this GPU.
- B3 mistral.rs logprobs layout: no existing issue found for the token/bytes mismatch. Related: #1876 (completions ignored logprobs, closed) and #2329 (open since July: greedy logprobs were the log of raw logits). #2329 does not apply to v0.9.5: `sample_argmax` applies softmax before `.ln()` (`sampler.rs:1165`), so it does not affect this study's first-token numbers.
- B4, B5, B7: no existing issue found.
- B6 mistral.rs throughput: closely related to the open issue #2343 ("Concurrent text-serving throughput does not scale above serial", v0.9.0 on an H100: 341 tok/s at 32 concurrent against 439 serial, 30 times lower than vLLM; opened 2026-07-12, no comments). Our tuning data shows the same shape on a 4090 (about 850 to 985 tok/s with defaults against 4583 for vLLM at similar load). This is the largest open lead: an unresolved performance problem that nobody has root-caused.
- Hypothesis 1 (vLLM slot swapping): no existing report found; open vLLM LoRA issues #63924 and #67617 concern pinned adapters and are different.
- Hypothesis 2 (SGLang prefix cache across adapters): SGLang already namespaces its radix cache by adapter (closed #24638, #9144), so no sharing across adapters looks deliberate. #49624 is about FlexKV isolation and is a different issue.

## Hypothesis 1 analysis from reading vLLM 0.34.0 source (not measured)

Result to explain: with 32 adapters and 16 slots, vLLM's last passing rate under uniform traffic is 10 rps, against 80 for no adapter.

- The scheduler caps distinct adapters per step: in `vllm/v1/core/sched/scheduler.py` (lines 1130 to 1139), a waiting request is skipped when `max_loras` adapters are already scheduled and its adapter is not one of them. So a step can serve at most 16 distinct adapters, and requests for the other 16 wait. A queued request gets in only when all running requests of some adapter finish. With 128-token outputs and uniform popularity, that turnover is slow, which would show up as large TTFT even though the GPU could do more work. This looks like a design trade-off in admission, not a copy cost.
- Activation of a newly needed adapter (`model_manager.py`, `activate_adapter`) runs synchronously in the engine loop and calls `set_lora` for every LoRA module (28 layers times 7 projections for this model), each with small non-blocking host-to-device copies. This is a second cost, per new adapter, and it is not overlapped with compute. Its size is unmeasured.
- Prediction to test: at `max_loras=32` the slowdown should mostly disappear (replication B). If it does, the finding is a configuration trade-off, not a bug. A profile is needed to split the queueing cost from the activation cost.
- Possible improvement if the activation cost matters: batch the per-module copies, or prefetch the adapter of the next waiting request while the current step runs. Unmeasured, so not a claim.

## B6 analysis so far (mistral.rs v0.9.5; source reading and the tuning-run server log; nothing measured on purpose yet)

- Default scheduler settings in the tuning log: `max_num_seqs=32`, `max_num_batched_tokens=4096`, `max_prefill_chunk_tokens=512`, `max_decode_steps_before_prefill=8`; with `--max-seqs 256` only the first changes. Prefix caching and CUDA decode graphs are on (18 graphs captured, batch bucket up to 32).
- The server's own throughput line in the saturated default run: 32 running, 0 waiting, prefill about 2,000 tok/s and decode about 930 tok/s. By Little's law that is about 7.5 completed requests per second and about 33 ms per generated token per request, including interleaved prefill, against about 5 to 7 ms for vLLM at similar load. So the engine, not the queue, is the limit.
- Ruled out by reading: the per-sequence CPU copy of the full logits for greedy sampling. For greedy requests without logprobs, penalties or logit bias, the batched CUDA sampling plan applies (`sampler.rs`, `cuda_batch_sampling_plan`), which is how our load generator asks.
- Not ruled out: slow prefill (2,000 tok/s is very low for this model; a prefill chunk of 512 tokens seems to take on the order of 100 ms or more), per-step CPU overhead in the engine loop, and the forced prefill step after every 8 decode steps.
- Test (needs the GPU, one short run at the default settings): while a rate-10 load runs, sample `nvidia-smi dmon` (utilization, SM clock, power) and the server's per-5-second throughput line. Low utilization with a slow step means CPU-side overhead or synchronization; high utilization with a slow step means inefficient kernels (prefill attention is the first thing to inspect). A second run with `max_decode_steps_before_prefill` changed would show whether the forced prefill cadence matters.
- Related open upstream issue: mistral.rs #2343 (no comments).

## B8. SGLang 0.5.23: the default `--sampling-defaults model` applies the model's generation_config on `/v1/chat/completions` but not on `/v1/completions` or the native `/generate` (differential study; found 2026-11-22)

(Numbered B8 because line 72 of this file already mentions a "B7" that has no entry; the differential plan had assumed B7.)

- Symptom: with `Qwen/Qwen2.5-1.5B-Instruct`, whose `generation_config.json` sets `repetition_penalty` 1.1, `top_p` 0.8, `top_k` 20 and `temperature` 0.7, the same request with only `temperature: 0` gives different text on SGLang's two OpenAI endpoints. 64 fixed prompts, `max_tokens` 128, no sampling parameters other than `temperature: 0`, server started with defaults (`--attention-backend` default, context 8192). Output classified against two transformers references (raw greedy with `repetition_penalty=1.0`, and Hugging Face's own default, which applies the model's penalty), text prefix-equal over the shorter length (at least 20 characters):

  | Endpoint | equals raw only | equals HF-default only | equals both | equals neither |
  |---|---|---|---|---|
  | SGLang `/v1/completions` | 34 | 0 | 10 | 20 |
  | SGLang `/v1/chat/completions` | 6 | 1 | 10 | 47 |
  | vLLM 0.34.0 completions / chat | 2 / 2 | 28 / 26 | 9 / 9 | 25 / 27 |
  | mistral.rs v0.9.5 completions / chat | 2 / 1 | 25 / 25 | 9 / 10 | 28 / 28 |

  vLLM and mistral.rs give the same behavior on both endpoints (HF-default-like); SGLang does not. A separate earlier run through SGLang's native `/generate` with `temperature: 0` only matched the raw reference on 50 of 64 prompts (identical to its explicit-neutral run), so `/generate` also does not apply the model defaults.
- Verified: one run per cell on SGLang 0.5.23 (RTX 4090, bf16), reproduced across two separate runs for the completions/generate side (the phase-1 C8 run and the endpoint run). Raw data: `differential/results/raw/axis1/` (C8) and the endpoint run on the pod (copied to `differential/results/raw/c8_endpoints/`). Not run on current `main`.
- Cause (read in source at tag v0.5.23): `OpenAIServingChat.__init__` sets `self.default_sampling_params = tokenizer_manager.model_config.get_default_sampling_params()` (`python/sglang/srt/entrypoints/openai/serving_chat.py:416-418`, logging "Using default chat sampling params from model generation config") and uses it at line 1334. `serving_completions.py` contains no reference to `default_sampling_params`, `sampling_defaults` or `generation_config`. `get_default_sampling_params` (`configs/model_config.py:2210-2240`) reads `repetition_penalty`, `temperature`, `top_k`, `top_p` and `min_p` when `sampling_defaults == "model"`; that is the server default (`arg_groups/fields/serving.py:232-238`), and the flag's help text ("Where to get default sampling parameters ... 'model' uses the model's generation_config.json ... Default is 'model'") does not limit it to chat.
- Also seen, not explained: on SGLang's chat endpoint the output matches neither reference for 47 of 64 prompts, while vLLM's chat output matches the HF-default reference on 35 of 64 (26 + 9). That suggests SGLang's repetition-penalty semantics differ from Hugging Face's and vLLM's. Not investigated.
- Defect or design choice: not settled. The log line says "chat", which suggests chat-only may be intentional; the flag's help text and the docs I read do not say so. vLLM documents applying the model's defaults on all its generation endpoints.
- Candidate fix (not written): apply `get_default_sampling_params()` in the completions handler as chat does, or state the scope in the flag's help text. The completions side looks small; the native `/generate` is a different question. Not tested.
- Update 2026-11-24 (pod clock 21:19, Item C): reproduced on SGLang `main` `0e9cff98f5b7` by server-level test, Qwen2.5-1.5B-Instruct, 64 prompts: (i) vs (ii) differ chat 19/64 (repeat noise 19/64, applied), `/v1/completions` 48/64 (noise 0, not applied), `/generate` 48/64 (noise 0, not applied). Regression test fails on `main`. Chat first pass differs from every later pass on 19/64 prompts (same 19; later passes agree, chat (ii) equals completions (ii) on 64/64); cause not yet tested. See `followup/pod_items/RESULTS.md`. Not posted; no fix written.
- Not yet verified: current `main`; whether chat-only is intended; other models with a `repetition_penalty` in `generation_config`; behavior at `temperature > 0`; why SGLang's chat output matches neither reference; a regression test.
- Upstream search (read-only, 2026-11-22; keyword queries, could miss things): no existing issue or PR found for the chat-versus-completions difference. Related open PRs: #45611 ("Apply --preferred-sampling-params on /v1/chat/completions", fixes #39096, created 2026-10-21) and #51858 ("Honor the Responses temperature default over HTTP"). #45611 describes the chat request resolving every sampler key through `get_param()` (user value, then model generation_config, then OpenAI defaults).
- Where it falls in the rubric: output differs by default for a common model (Qwen2.5) with no error, so it belongs in the silent-difference class if it turns out to be unintended; it is dropped as a design choice if SGLang documents it as chat-only.

## B6 profile result (2026-11-22, pod clock 08:25 UTC; second machine, RTX 4090, mistral.rs v0.9.5 rebuilt, default scheduler settings, 8 adapters loaded)

Closed loop, concurrency 16, 220 requests per run, uniform choice among N adapters, Qwen2.5-1.5B-Instruct. Time per output token (median) and tokens per second:

| distinct adapters in the mix | ms per token | tok/s | errors of 220 |
|---|---|---|---|
| 0 (base model) | 10.3 and 10.6 (two runs) | 1462 and 1431 | 0 |
| 1 | 16.6 and 16.4 | 897 and 906 | 1 |
| 2 | 17.4 | 863 | 1 |
| 4 | 20.9 | 720 | 1 |
| 8 | 24.8 and 22.0 | 606 and 703 | 2 and 3 |

- Using any adapter costs about 6 ms per token at once (about 60% more), and each further adapter adds roughly 1 to 2 ms: close to linear growth in the number of adapters per batch.
- GPU utilization while active was higher with adapters (mean SM busy 71 to 75%, against 57% for the base model) while power draw was not higher (160 to 220 W against 188 W). That fits many small kernels more than heavier compute, but is not proof.
- Requests with errors ("no tokens") appeared only when adapters were used: 1 to 3 of 220.
- Hypotheses to test, from reading `mistralrs-quant/src/lora/dynamic/cuda.rs`: the code launches work per active adapter slot (so launches per step grow with adapters times modules times layers), and the decode path may not use the captured CUDA graphs when adapters are active. Test: count kernel launches per decode step with 1 and 8 adapters (Nsight Compute can count them), and check whether graph replay is used.


## B9. `runpod2`: CUDA driver init fails (`cuInit` returns 999, CUDA_ERROR_UNKNOWN) after about 07:14 UTC on 2026-11-24 (infrastructure, not an engine bug)

- Symptom: `nvidia-smi` works (RTX 4090, 1 MiB used, P8, no compute processes), but `cuInit(0)` returns 999 through `libcuda.so.1`, so `torch.cuda.is_available()` is False in every environment (vllm, sglang) with "CUDA unknown error". The first sign was vLLM dying at warm-up with "CUDA error: unknown error" inside flashinfer (07:15:57), after the GPU had worked normally for a SGLang `main` check at 07:09 to 07:12 and for the 3a runs earlier. `nvidia-smi --gpu-reset` answers "Not Supported" in this container.
- Not known: the cause. A concurrent mistral.rs build (nvcc, CPU only) and a vLLM start with `CUDA_HOME` set (a script mistake, fixed, see CHANGELOG) were running around the time; neither is shown to be the cause. A container with an abnormally killed GPU process (`pkill -9`) is another candidate; not tested. Driver 595.112.07, open kernel module.
- Upstream search: not done (this looks like a host or driver state, not engine code); if it recurs on the dedicated machine, search NVIDIA forums for `cuInit 999` with the driver version.
- Effect: Item B, the 3d reproduction, the 3e experiment and the dry run of `run_session.sh` need the GPU and are blocked until it recovers (a pod restart by the author may be needed; `/dev/shm` contents, the adapters and the model copy there, would be lost and regenerated from the scripts). CPU-only work continues.

## Item D (3e): mistral.rs first-use difference, not reproduced (2026-11-24, pod clock 21:31 to 22:22)

Not a bug found. Lead: in the kernel rerun's 128-completion check the first prompt with each adapter varied between launches. Test: 6 starts, old and new trees alternating, serial identical requests (3 base, 6 per adapter, 3 more per adapter), 8 adapters, `/v1/completions`, `temperature` 0. Result: 0 of 9 cells differ between request 1 and later requests in any start; one output per cell across all starts and both trees. The earlier variation (8 adapters in flight together after a restart) is not explained and was not tested here. See `followup/pod_items/RESULTS.md`, Item D.

