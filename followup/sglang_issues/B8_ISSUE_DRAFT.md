# B8 issue draft (NOT posted; for the author to review, 2026-11-24)

## 2d findings (read-only, SGLang `main` at 5d14e90, 2026-11-24 02:55 UTC)

- Still chat-only on `main` by source: `serving_chat.py` reads `model_config.get_default_sampling_params()` (lines 424 to 434) and passes it as `model_generation_config=` at line 1267; `serving_completions.py` has no reference to `default_sampling_params`, `sampling_defaults` or `generation_config`, and its `_build_sampling_params` (line 148) copies `request.temperature`, `request.top_p`, `request.top_k`, `request.repetition_penalty` and the other fields directly. Confirmed by running a server on `main` on 2026-11-24 (Item C, below).
- Intent: the flag was added in #11304 (merged 2025-10-08, by a collaborator), titled "[oai serving chat] Add argument `--sampling-defaults` and fix `ChatCompletionRequest` defaults". Its text speaks only about `ChatCompletionRequest` and says nothing for or against other endpoints. The log line says "chat". Neither shows that chat-only is intended; they show the change was written for chat. `git blame` was not needed beyond this: `serving_completions.py` commits on main touching sampling or defaults are #38049 (config refactor), #16711 (usage flag) and #15498 (seed), none about generation_config.
- Overlap, checked 2026-11-24:
  - #45611 (open): `--preferred-sampling-params` on chat. A different flag and path; same family of problem.
  - #51858 (open): the Responses temperature default over HTTP in the Rust server. Not the same.
  - #52665 (open, opened 2026-11-23, contributor): applies `--preferred-sampling-params` on `/v1/completions` and `/v1/responses`. It does not mention `--sampling-defaults model` or the model's `generation_config` on completions, so it does not fix B8 as written, but a fix to completions sampling would touch the same function and should be coordinated with it.
  - #15764 (closed, not merged): preferred params on chat.
  No existing issue found for the `--sampling-defaults model` chat versus completions difference (keyword search, can miss).

## Issue text (draft; revised 2026-11-24 after the server-level reproduction on `main`; NOT posted)

**Title:** `--sampling-defaults model`: the model's generation_config is not applied on /v1/completions or /generate (main 0e9cff98f5b7)

With the server default `--sampling-defaults model`, a request to `/v1/completions` that sets `temperature: 0` gets different output from the same request with the model's own `generation_config.json` values sent explicitly. On `/v1/chat/completions` the two give the same output. Is it intended that `/v1/completions` does not apply the model's defaults? I could not find a statement either way; the flag was added in #11304 for `ChatCompletionRequest`.

Setup: SGLang `main` at `0e9cff98f5b7` (includes #46500), one RTX 4090, `Qwen/Qwen2.5-1.5B-Instruct` (`generation_config.json`: `repetition_penalty` 1.1, `top_p` 0.8, `top_k` 20, `temperature` 0.7), no LoRA, server flags default except `--context-length 8192`. 64 fixed prompts, 128 new tokens, one request at a time. Request (i) sets `temperature: 0` alone. Request (ii) sets `temperature: 0` and `repetition_penalty: 1.1`, `top_p: 0.8`, `top_k: 20`. For each endpoint I also repeated (i) to measure run-to-run noise. For `/v1/completions` and `/generate` the prompt is the chat-templated text, so all three endpoints see the same tokens.

| Endpoint | (i) differs from (ii) | (i) differs from its repeat |
|---|---|---|
| `/v1/completions` | 48 of 64 | 0 of 64 |
| `/generate` | 48 of 64 | 0 of 64 |
| `/v1/chat/completions`, repeat pass | 0 of 64 | not applicable |

On `/v1/completions`, (i) differs from (ii) on 48 prompts while repeats of (i) agree on all 64, so the difference is not noise. I read this as the model's defaults not being applied. `/v1/completions` (ii) equals `/generate` (ii) and `/v1/chat/completions` (ii) on 64 of 64 prompts.

On `/v1/chat/completions`, the repeat of (i) gives the same text as (ii) on 64 of 64 prompts, and the server logs "Using default chat sampling params from model generation config: {'repetition_penalty': 1.1, 'temperature': 0.7, 'top_k': 20, 'top_p': 0.8}". So chat applies the defaults. Separately, the initial pass on a freshly started server differed from every later pass on 19 of 64 prompts; With the radix cache disabled (`--disable-radix-cache`) the difference was absent (0 of 64), so it looks like a cache-state effect; I do not claim it is related to this issue.

Question for the maintainers: is `/generate` meant to ignore the model's defaults? It behaves like `/v1/completions` here, and I could not tell whether that is deliberate.

Source (by reading `main` at `0e9cff98f5b7`): `serving_chat.py` reads `model_config.get_default_sampling_params()`; `serving_completions.py` has no reference to it, and its `_build_sampling_params` copies the request fields directly.

Regression test (needs no GPU or model): `test_completions_sampling_defaults.py` builds `OpenAIServingCompletion` with a stub whose `get_default_sampling_params()` returns the three values, calls `_build_sampling_params` for a request that sets `temperature` alone, and asserts the model values are used. Output on `main` at `0e9cff98f5b7`:

```
FAIL test_model_defaults_apply_when_request_sets_only_temperature -> AssertionError 1.0
PASS test_request_value_wins_over_model_default
PASS test_schema_defaults_when_model_has_none
```

The failure is the assertion on `repetition_penalty` (1.0 where the model says 1.1). I have not written a fix.

Not known: other models; behavior at `temperature > 0`; whether #52665 (open, adds `--preferred-sampling-params` on `/v1/completions`) changes this; whether `/generate` should follow chat. Data and scripts are in the study repo (link a commit before posting): `followup/pod_items/results/C_repro/`, `followup/pod_items/cross_endpoint.py`. An earlier comparison on v0.5.23 against vLLM 0.34.0 and mistral.rs v0.9.5 showed the same chat versus completions difference; vLLM and mistral.rs behave the same on both endpoints.

AI tools helped with analysis scripts and code reading. The measurements and conclusions are mine.

## Sketch of a completions-side fix (not written, not tested)

In `OpenAIServingCompletion.__init__`, read `self.default_sampling_params = tokenizer_manager.model_config.get_default_sampling_params()` as chat does. In `_build_sampling_params`, for each of `temperature`, `top_p`, `top_k`, `min_p`, `repetition_penalty` that is not in `request.model_fields_set` and is present in `self.default_sampling_params`, use the model's value instead of the request schema default. Fields the client sets stay as they are. Same precedence as chat: request, then model, then built-in. Coordinate with #52665, which edits the same function.

Regression test (fails before the fix): build a `OpenAIServingCompletion` with a fake `model_config` whose `get_default_sampling_params()` returns `{"repetition_penalty": 1.1, "top_p": 0.8, "top_k": 20}`; call `_build_sampling_params` with a request that sets only `temperature`; assert the three values come from the model. Add a second case: the request sets `top_p` explicitly, assert the request value wins. A third case with `--sampling-defaults openai` asserts the schema defaults. Needs no GPU.
