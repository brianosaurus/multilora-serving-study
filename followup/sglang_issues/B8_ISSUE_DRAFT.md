# B8 issue draft (NOT posted; for the author to review, 2026-11-24)

## 2d findings (read-only, SGLang `main` at 5d14e90, 2026-11-24 02:55 UTC)

- Still chat-only on `main` by source: `serving_chat.py` reads `model_config.get_default_sampling_params()` (lines 424 to 434) and passes it as `model_generation_config=` at line 1267; `serving_completions.py` has no reference to `default_sampling_params`, `sampling_defaults` or `generation_config`, and its `_build_sampling_params` (line 148) copies `request.temperature`, `request.top_p`, `request.top_k`, `request.repetition_penalty` and the other fields directly. Not yet confirmed by running a server on `main` (that is item 3d, GPU).
- Intent: the flag was added in #11304 (merged 2025-10-08, by a collaborator), titled "[oai serving chat] Add argument `--sampling-defaults` and fix `ChatCompletionRequest` defaults". Its text speaks only about `ChatCompletionRequest` and says nothing for or against other endpoints. The log line says "chat". Neither shows that chat-only is intended; they show the change was written for chat. `git blame` was not needed beyond this: `serving_completions.py` commits on main touching sampling or defaults are #38049 (config refactor), #16711 (usage flag) and #15498 (seed), none about generation_config.
- Overlap, checked 2026-11-24:
  - #45611 (open): `--preferred-sampling-params` on chat. A different flag and path; same family of problem.
  - #51858 (open): the Responses temperature default over HTTP in the Rust server. Not the same.
  - #52665 (open, opened 2026-11-23, contributor): applies `--preferred-sampling-params` on `/v1/completions` and `/v1/responses`. It does not mention `--sampling-defaults model` or the model's `generation_config` on completions, so it does not fix B8 as written, but a fix to completions sampling would touch the same function and should be coordinated with it.
  - #15764 (closed, not merged): preferred params on chat.
  No existing issue found for the `--sampling-defaults model` chat versus completions difference (keyword search, can miss).

## Issue text (draft)

**Title:** `--sampling-defaults model` applies generation_config on /v1/chat/completions but not on /v1/completions (v0.5.23)

With the server default `--sampling-defaults model`, the model's `generation_config.json` is used for requests to `/v1/chat/completions` but not `/v1/completions` or `/generate`. Is chat-only intended? The flag was added in #11304 for `ChatCompletionRequest`; I could not find a statement about other endpoints.

Setup: SGLang 0.5.23, one RTX 4090, `Qwen/Qwen2.5-1.5B-Instruct` (`generation_config.json`: `repetition_penalty` 1.1, `top_p` 0.8, `top_k` 20, `temperature` 0.7). 64 fixed prompts, `max_tokens` 128, the request sets only `temperature: 0`, server started with defaults. Each output is compared with two transformers references (raw greedy with `repetition_penalty=1.0`; and Hugging Face's own default, which applies the model's penalty), by text prefix over the shorter length:

| Endpoint | equals raw only | equals HF-default only | equals both | equals neither |
|---|---|---|---|---|
| SGLang `/v1/completions` | 34 | 0 | 10 | 20 |
| SGLang `/v1/chat/completions` | 6 | 1 | 10 | 47 |
| vLLM 0.34.0 completions / chat | 2 / 2 | 28 / 26 | 9 / 9 | 25 / 27 |
| mistral.rs v0.9.5 completions / chat | 2 / 1 | 25 / 25 | 9 / 10 | 28 / 28 |

vLLM and mistral.rs behave the same on both endpoints. SGLang's completions output looks like the raw reference; its chat output matches neither reference for 47 of 64 prompts, which may mean SGLang's repetition penalty differs from Hugging Face's; I did not look into that.

Source (v0.5.23, and `main` at 5d14e90 by reading): `serving_chat.py` uses `get_default_sampling_params()`; `serving_completions.py` does not.

Not known: whether chat-only is deliberate; behavior with `temperature > 0`; other models; whether #52665 changes this. Not tested on `main` yet. Data and scripts: `differential/results/raw/axis1/` and `differential/results/raw/c8_endpoints/` in the study repo (link to a commit before posting).

AI tools helped with analysis scripts and code reading. The measurements and conclusions are mine.

## Sketch of a completions-side fix (not written, not tested)

In `OpenAIServingCompletion.__init__`, read `self.default_sampling_params = tokenizer_manager.model_config.get_default_sampling_params()` as chat does. In `_build_sampling_params`, for each of `temperature`, `top_p`, `top_k`, `min_p`, `repetition_penalty` that is not in `request.model_fields_set` and is present in `self.default_sampling_params`, use the model's value instead of the request schema default. Fields the client sets stay as they are. Same precedence as chat: request, then model, then built-in. Coordinate with #52665, which edits the same function.

Regression test (fails before the fix): build a `OpenAIServingCompletion` with a fake `model_config` whose `get_default_sampling_params()` returns `{"repetition_penalty": 1.1, "top_p": 0.8, "top_k": 20}`; call `_build_sampling_params` with a request that sets only `temperature`; assert the three values come from the model. Add a second case: the request sets `top_p` explicitly, assert the request value wins. A third case with `--sampling-defaults openai` asserts the schema defaults. Needs no GPU.
