# Differential study of vLLM, SGLang and mistral.rs: pre-registered plan

> **Status: DRAFT for Gate 1. No measured run has happened.** Drafted by Claude (AI) on 2026-11-22; not reviewed by Brian line by line. Facts about engine flags and support come from AI subagents reading docs and source at the pinned tags (marked *(agent-read)*) and are verified on the machine before use. Local only: nothing here is pushed, filed or posted.

## 1. Question

Where do the three engines, at their current pinned releases, differ significantly from a reference, and which difference is the most severe? Four axes, each against its own fixed reference:

| Axis | Reference |
|---|---|
| 1. Output correctness | Hugging Face transformers, bf16 |
| 2. API conformance | The OpenAI chat and completions API specification |
| 3. Performance | The GPU's own measured limits (HBM bandwidth, dense BF16 GEMM) |
| 4. Robustness | The expected behavior: clear 4xx, no crash, no leak |

This is a differential study of three engines on one GPU. It does not claim to be first, only or comprehensive.

## 2. Prior work (read by an AI agent from arXiv HTML and README pages; not yet checked line by line by Brian)

| Work | What it covered | Not covered |
|---|---|---|
| **arXiv 2605.19537**, "The Silent Hyperparameter: Quantifying the Impact of Inference Backends on LLM Reproducibility" (Pape, Evertz, Schönherr; v4 2026-11-18). Code: github.com/Dormant-Neurons/inference_backends | vLLM 0.10.2, SGLang 0.5.2, llama.cpp, LMDeploy, Ollama vs transformers 4.57.0; Llama 3.1 8B, Qwen3 4B/30B, DeepSeek-R1-Distill-Qwen 7B; H100; greedy, batch 1, 12 seeds; task accuracy spread up to 16.6 points, disagreement rate, logprob RMSE; a few ablations (prefix cache, CUDA graphs, BOS token, repetition penalty); throughput/TTFT only in an appendix | API conformance, robustness, mistral.rs, any hardware-ceiling analysis, a systematic cell matrix (chunked prefill, stop, batch), current versions |
| **infer-check** (github.com/NullPointerDepressiveDisorder/infer-check, PyPI 0.2.x) | A CLI: quantization sweeps, cross-backend compare, determinism, stress; tested on Apple Silicon with mlx-lm; generic OpenAI-compatible backend | No CUDA vLLM/SGLang backends (per its README), no API conformance, no mistral.rs, no versions given |
| **GRIEF**, arXiv 2605.11202 | Fuzzing of vLLM 0.18/0.19 and SGLang 0.5.10.post1 (Qwen2.5-0.5B, H100) with a logprob-replay oracle inside one engine; state corruption, performance pathology, crash/liveness | A reference implementation, API field conformance, hardware-ceiling performance, mistral.rs; whether it tests disconnect-frees-KV is not verified |
| **Ekka**, arXiv 2606.04594 (ICML 2026) | An empirical study of 90 silent-error issues from vLLM/SGLang and a diagnosis tool using a transformers reference; 4 new bugs | It diagnoses known failures; no cell sweep, no API conformance, no mistral.rs |

Nothing here is claimed as new beyond its own scope: a per-cell comparison on current releases that includes mistral.rs, API conformance, roofline-relative performance and robustness. Earlier work may already cover parts of it.

## 3. Machine and rules

- **Machine:** `ssh runpod2`, hostname **`bc2389eb52de`** (RTX 4090 24 GB, sm_89, driver 595.112.07). Brian confirmed this as the machine for this follow-up on 2026-11-22; the multi-LoRA notes record a different hostname for the `runpod` alias, so the hostname is re-checked before **every** session and every script (scripts exit on mismatch).
- **One engine process on the GPU at a time.** `nvidia-smi` idle check (<500 MiB used) before every run, recorded. No run while another study is measuring (checked: nothing running at plan time).
- **Pinned versions** (already installed, reused, not upgraded): vLLM 0.34.0 (`62d4bc3c7e33`) and SGLang 0.5.23 (`297e54fd105e`) in `/root/kvc/venv-vllm` and `/root/kvc/venv-sglang`; mistral.rs v0.9.5 prebuilt CUDA 12.8 sm89 binary (commit `465191d0b0a7`, sha256 verified). A **new** venv for the reference (transformers, torch, jsonschema, etc.); engine venvs are not modified.
- **SGLang JIT toolchain:** SGLang's JIT must use the CUDA 13.5 nvcc shipped in its venv (`CUDA13` shim from the kv-conformance work), not `CUDA_HOME=/usr/local/cuda` (12.8); otherwise HiCache-class kernels build with mismatched headers. Recorded per run.
- **Storage:** weights read during runs come from `/dev/shm` (RAM-backed, 19 GB, noexec); one model at a time. Cold copies of the two larger models live on the `/workspace` network volume (quota ~47 GiB) and are copied to `/dev/shm` before use; never read from `/workspace` during a run. Local disk has ~5.6 GB free; I will list what I propose to delete (only files this work created) and ask before deleting.
- **Nothing leaves Brian's machines** without exact text shown first. No pushes, issues, comments, PRs or posts. Look before deleting. Negative results reported plainly.
- **Every run records:** full command line, versions, commit hashes, GPU state (clocks, temperature), storage layout. Raw results are kept and never edited.

## 4. Models

| Role | Repo (ungated, via HF API without a token) | Params / bf16 size | Notes |
|---|---|---|---|
| Qwen | `Qwen/Qwen2.5-1.5B-Instruct` | 1.54B / 3.1 GB | Used in the multi-LoRA study |
| Llama family | `unsloth/Llama-3.2-3B-Instruct` | 3.21B / 6.4 GB | Third-party mirror of meta-llama; weights not byte-compared to the original (gated, no token); recorded as a limitation |
| Mistral family | `unsloth/mistral-7b-instruct-v0.3` | 7.25B / 14.5 GB | Mirror; alternates `mistral-community/...`, `mistralai/Mistral-7B-Instruct-v0.3`. Leaves ~8 GB for KV on 24 GB |

Architectures `LlamaForCausalLM`, `MistralForCausalLM`, `Qwen2ForCausalLM` are registered in all three engines *(agent-read; each model's config and tokenizer/chat template are verified at serve time, and any engine that cannot serve a model is recorded as "unsupported" for that model)*.

## 5. Axis 1: output correctness vs transformers bf16

**Reference.** transformers (pinned in the new venv; version recorded), bf16, sdpa attention, batch size 1 (no padding), greedy, 128 new tokens, stop only at EOS or length. Per prompt the reference stores: prompt token ids, generated token ids, and for each generated token the log-softmax (fp32 over bf16 logits) of the chosen token and the top-20 from a separate teacher-forced forward pass over prompt+continuation. A repeat of the reference gives the reference's own run-to-run spread (expected 0; recorded).

**Prompts.** 64 fixed prompts, built by a committed script with a fixed seed and no external data: 16 short factual questions, 16 arithmetic/reasoning, 16 code, 16 passage-based (≈300-token passage plus a question). A second set of 16 "trivially short" prompts (1 to 4 tokens) is used for the noise floor. The prompt files and their sha256 are committed before any engine run. Chat prompts are rendered with each model's own chat template by the reference tokenizer; the engines are sent **token ids** where they accept them (vLLM, SGLang completions); where an engine only takes text, the text is the detokenized id sequence and `usage.prompt_tokens` must equal the reference length, otherwise that prompt is excluded for that engine and counted.

**Metric.** Teacher-forced: each engine is given the reference's prompt and its 128 generated tokens and returns the logprob of each of those tokens. Per prompt, `d = mean over 128 positions of |logprob_engine − logprob_ref|`. The cell statistic `D` is the mean of `d` over the 64 prompts with a 95% bootstrap CI (10,000 resamples over prompts). Also recorded: the fraction of positions where the engine's top-1 differs from the reference token and the reference's gap there (near-tie handling as in the kv-conformance study). Methods per engine *(agent-read, to be verified)*: vLLM `prompt_logprobs` (completions, `echo`+`max_tokens` small); SGLang `/generate` with `return_logprob`, `logprob_start_len`; **mistral.rs has no `prompt_logprobs` and `echo` logprobs are unverified**: if teacher forcing is unavailable there, its fallback is generated-token logprobs compared up to the first divergence from the reference, with the divergence position recorded, and its results are labeled as a different method and not pooled with the others.

**Noise floor (measured first, per engine and model, defaults, batch 1):**
- `R_e`: run-to-run spread = `D` between two identical runs of the same cell on the same server (5 repeats; max pairwise).
- `S_e`: the engine's own `D` vs the reference on the 16 trivially short prompts (32 teacher-forced tokens; prefill numerics are negligible there, so this is decode-kernel noise).
- `N_e = max(R_e, S_e)`, floored at 0.005 nats.

**Flag rule (pre-registered).** A cell is flagged if `D_cell > 3 × N_e` **and** `D_cell − N_e > 0.02` nats **and** the lower end of the bootstrap CI of `D_cell` exceeds `1.5 × N_e`. A flagged cell must reproduce in at least 2 of 3 repeated runs to count as a difference. Greedy-generation divergence (the engine generating its own tokens) is recorded separately: first-divergence position and the reference's gap there.

**Cells** (each is one pass of the 64 prompts per engine and model unless noted; "defaults" means each engine's documented defaults plus the flags needed to serve the model):

| Cell | What varies |
|---|---|
| C0 | Baseline: greedy, sequential (batch 1), defaults |
| C1 | Sampling path: temperature 0.7, per-request seed fixed. Checks (a) the same seed run twice gives identical output, (b) the mean reference NLL of the engine's sampled tokens under temperature-0.7 reference logits vs transformers' own samples (same seeds): difference > 0.05 nats/token with a CI excluding 0 is flagged |
| C2 | Stop sequences: single, multi-token, 4 strings, a string present at token 1, one absent, one split across tokens. Pass = output text equals the reference text cut at the first occurrence (stop string excluded), `finish_reason == "stop"` |
| C3 | `max_tokens` edges: 0 (where allowed), 1, 2, the reference's EOS position, exactly the remaining context, one more than the context. Pass = spec behavior (length vs error), token counts and `finish_reason` correct |
| C4 | Long prompts: 8 prompts each at 1,024, 2,048, 4,096 tokens, plus the boundary: prompt = limit−128 (generate to the limit), prompt = limit (must be rejected or truncated per docs), where the context limit is 8,192 set on every engine that allows it (mistral.rs's `--max-model-len` is rejected by the Qwen2 loader; its native limit is used and recorded) |
| C5 | Chunked prefill off vs on at 4,096-token prompts. vLLM `--[no-]enable-chunked-prefill`; SGLang `--chunked-prefill-size -1` vs default; mistral.rs has no off switch, so `--max-prefill-chunk-tokens` 128 vs 4,096 is the analogue (labeled as such) |
| C6 | Prefix cache on vs off, each prompt sent twice (the second is a hit). vLLM `--[no-]enable-prefix-caching`; SGLang `--disable-radix-cache`; mistral.rs `--prefix-cache-n 0`. A hit counts only if the engine's own counter or `cached_tokens` confirms it; a miss is **invalid, never a pass** |
| C7 | Concurrency: the same 64 prompts at batch 1 (sequential) vs 32 concurrent |

Server configurations per engine and model: default, no-chunked-prefill/analogue, no-prefix-cache. With 3 engines × 3 models that is at most 27 server starts for Axis 1, roughly 1 to 2 hours including the 7B model.

## 6. Axis 2: API conformance vs the OpenAI specification

**Reference:** the OpenAI OpenAPI specification (github.com/openai/openai-openapi, a specific commit recorded at the start of the axis) plus the OpenAI API reference pages for chat completions and completions. Where the spec is ambiguous, the observation is recorded as "ambiguous", not as a defect.

**Checks** (each run on every engine, both endpoints where they exist; every difference is recorded with the exact request and the raw response):
1. **logprobs:** shape of `logprobs` on `/v1/completions` (`tokens`, `token_logprobs`, `top_logprobs`, `text_offset`) and on chat (`content[].token/logprob/bytes/top_logprobs`); `top_logprobs` size; token strings vs ids; values compared to the reference (links to Axis 1).
2. **Streaming:** `data:` framing, `[DONE]`, object type `chat.completion.chunk`, role in the first chunk, constant `id`, `finish_reason` in the last content chunk, `stream_options.include_usage` produces a final usage chunk with an empty `choices` list (mistral.rs documents ignoring `stream_options`: recorded as a documented deviation).
3. **`n > 1`:** choices indexed 0..n−1, distinct with temperature > 0, usage sums.
4. **`stop`:** string vs list, up to 4 sequences, more than 4 rejected.
5. **usage:** `prompt_tokens` equals the length from the reference tokenizer (chat template applied) and `completion_tokens` equals the number of generated tokens (EOS counting recorded, not judged); `total_tokens` sum.
6. **`finish_reason`:** `stop`, `length`, `tool_calls` as appropriate.
7. **Errors:** malformed JSON, missing required field, wrong types, out-of-range values (temperature 2.1, `n=0`, `max_tokens=-1`), empty `messages`, unknown model. Expected: HTTP 4xx with the OpenAI error body `{error:{message,type,param,code}}`; a 5xx or a crash is recorded.
8. **`response_format` `json_schema`** (strict): 20 fixed schemas; the output must parse and validate (jsonschema); `json_object` also tested (mistral.rs documents rejecting it).
9. **`tools` and `tool_choice`** (`auto`, `required`, `none`, named function): `tool_calls` shape, arguments valid JSON validating against the schema. Tool parsing flags that the engine documents as required are used and listed.

Each check yields: spec-conformant / differs (what field) / unsupported-and-rejected-cleanly / accepted-and-silently-ignored / crash or 5xx. "Differs" items are counted by severity in the ranking.

## 7. Axis 3: performance vs measured hardware limits

**Ceilings, measured once and recorded with the GPU clocks and temperature:**
- HBM bandwidth: a large-tensor device copy and a read-reduce kernel (torch), the best of repeated runs, reported next to the spec value (1,008 GB/s) *(spec value to be verified)*.
- Dense BF16 GEMM: `torch.matmul` on large square matrices, best of repeated runs (TFLOP/s), reported next to the spec value.

**Metrics.**
- Batch-1 decode: tokens/s at 1 concurrent request, 256-token output, 128-token prompt. Achieved fraction of the bandwidth ceiling = `tokens/s × bytes read per token ÷ bandwidth`, where bytes per token = bf16 weight bytes + KV bytes read at that context.
- Prefill: tokens/s on 2,048-token prompts (generate 1 token). Achieved fraction of the compute ceiling = `tokens/s × 2 × params ÷ GEMM TFLOP/s` (attention FLOPs recorded separately).

**Same documented tuning budget for every engine** *(settings agent-read; verified before use)*. Defaults arm: each engine's documented defaults (+ flags needed to serve: mistral.rs `--kv-cache` where its default memory sizing fails on this GPU, see mistral.rs #2502 fixed after v0.9.5), with matched maximum concurrent sequences (32), matched context length where allowed, and a matched KV budget. Tuned arm: 6 server configurations per engine and model, chosen from these documented knobs only, each measured with the same 2-minute protocol, best selected on a separate selection run and then re-measured 3 times:
- vLLM: `--max-num-batched-tokens` {2048, 8192, 16384}, `--max-num-seqs` {32, 128}, CUDA graph mode, `--gpu-memory-utilization`, `--async-scheduling`.
- SGLang: `--chunked-prefill-size` {2048, 8192}, `--attention-backend` (documented backends), `--cuda-graph-max-bs-decode`, `--mem-fraction-static`, `--schedule-policy`.
- mistral.rs: `--max-prefill-chunk-tokens` {512, 2048, 8192}, `--max-num-batched-tokens`, `--pa-block-size` {16, 32}, `--max-seqs`, `--paged-attn`.

**A performance gap counts only if it survives tuning:** an engine's tuned fraction of the ceiling is below 0.8 × the best engine's tuned fraction on the same metric and model, with non-overlapping min-max ranges over 3 repeats. Gaps present only in the defaults arm are recorded as "defaults only" (lowest severity).

## 8. Axis 4: robustness

For each engine (documented configuration, Qwen2.5-1.5B), each check repeated 3 times:
1. **Unknown model name and unknown adapter name** (addressed the way each engine documents adapters): expect a clear 4xx with an error body; no crash.
2. **Malformed JSON, wrong content type, empty body.**
3. **`max_tokens` larger than the context:** expect a 4xx or documented clipping.
4. **Client disconnect mid-stream:** open a long streaming request, close the socket after 5 tokens; watch the engine's own gauges (vLLM `vllm:num_requests_running`, `vllm:kv_cache_usage_perc`; SGLang `sglang:num_running_reqs`, `sglang:kv_used_tokens`, `sglang:token_usage` with `--enable-metrics`; mistral.rs `mistralrs_sequences_running`, `mistralrs_kv_cache_blocks_used`, `mistralrs_sequences_completed_total{reason="canceled"}`). Pass = running count returns to 0 and KV usage to its idle value within 10 s. Repeat 200 times in a loop to detect a slow leak. Recorded also: GPU memory before and after.
5. **Overload past saturation:** closed-loop clients ramped to 4× each engine's max concurrent sequences with long generations for 60 s. Record: requests by outcome (completed, 429/503 with a JSON body, timeout, connection reset), whether the server stays up, latency recovery within 30 s after the load stops, and GPU/host memory growth. A clean queue, clean rejection or documented timeout passes; a crash, a hang, or growing memory fails.

## 9. Ranking (pre-registered rubric)

For each difference record: **defect or design choice** (cite the engine's docs, the spec or the reference; design choices and already-fixed or already-claimed items are dropped), **who hits it** (defaults? common models? standard fields?), **severity order**: silent wrong output > crash/500 > wrong API shape > slowdown that survives tuning > slowdown only on defaults, **fix size and files** (read in source vs confirmed at runtime), **upstream status** (search open and closed issues and PRs first; links recorded). Top 5 with evidence go to Brian at Gate 3. He chooses which to fix.

## 10. Gates

- **GATE 1 (this document):** exact cells, references, noise-floor procedure, thresholds, tuning budget. No measured run before Brian's review.
- **GATE 2:** all four axes run; a table of every significant difference: engine, axis, cell, size vs threshold, number of reproductions.
- **GATE 3:** the ranked list and the top 5 with evidence. Stop.
- After Brian chooses: a minimal reproduction independent of the harness; root cause with file and line; the fix's effect estimated *before* measuring; the fix on a branch with a regression test that fails before and passes after; the axis-1 cells for that engine rerun to confirm nothing else changed; draft issue and PR text for Brian's review. Nothing pushed or filed.

## 11. Logging

Bugs go to `docs/BUGS.md` of this repository, continuing its numbering (the highest existing entry is **B6**, so new entries start at B7), in its existing format, **the moment they are observed**, before investigating. Each entry has an upstream-search section. Every change to this plan after the first measured run goes to `differential/CHANGELOG.md` with the date and reason.

## 12. Risks and limits stated in advance

- The Llama and Mistral weights come from third-party mirrors (the originals are gated; no token); not byte-compared.
- Teacher forcing may be unavailable on mistral.rs; its Axis-1 results then use a different method and are not pooled.
- 24 GB limits the 7B model's KV budget and concurrency; the overload test at 4× max concurrency may need smaller contexts for it.
- One GPU, one machine, one run per tuned configuration in the tuning search (3 repeats on the winner); results do not generalize to other hardware.
- The pinned engine venvs also contain unrelated packages from earlier work (LMCache in the vLLM venv); they are not used here.
- AI wrote this plan and will write the harness; Brian has not reviewed either line by line.
