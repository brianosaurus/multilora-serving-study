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
- Not verified: whether `master` differs; whether it is deliberate; whether an issue exists. The request types are in `mistralrs-server-core/src/openai.rs` and the handler in `completions.rs`.
- Candidate fix: match the OpenAI field meanings (and the legacy completions layout for that endpoint), with a test.

## B4. mistral.rs: `--max-model-len` is listed by `serve --help` but rejected by the plain safetensors loader

- Symptom: `Error: max_model_len=4096 is not supported by this model loader` for a plain safetensors model.
- Verified on v0.9.5. May be intended; low priority. A clearer message or documentation of which loaders support it would help.

## B5. SGLang packaging: 0.5.23 needs a pre-release dependency

- `sglang==0.5.23` depends on `flash-attn-4>=4.0.0b18`, a pre-release. `uv pip install sglang==0.5.23` fails to resolve until that package is named explicitly. Plain `pip` may behave differently, so this may be a tooling difference and not a defect. Low priority.

## Planned exploratory probes

These run only when no measurement is in progress, because the GPU must be used by one process at a time. They are exploratory and are reported separately from the measured results: adapter swap storms with more adapters than slots (memory growth, errors, crashes); concurrent mixed-adapter load for cross-adapter contamination; error handling for an unknown adapter name, a wrong rank, and malformed requests (expect a client error, not a crash or a 500); long-run stability (memory growth, error rate over time); and repeated identical requests for determinism.
