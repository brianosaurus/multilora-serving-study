"""Regression test for B8 (plan: followup/pod_items/PLAN.md, Item C). Runs without a GPU or a model.
The completions handler of SGLang should use the model's generation_config defaults for sampling fields the request does not set,
as the chat handler does. Written against SGLang main 0e9cff98f5b7; case 1 is expected to FAIL there (that is the bug), cases 2 and 3 to pass.
Run: python test_completions_sampling_defaults.py   (also collectable by pytest)"""
from unittest.mock import MagicMock

from sglang.srt.entrypoints.openai.protocol import CompletionRequest
from sglang.srt.entrypoints.openai.serving_completions import OpenAIServingCompletion

MODEL_DEFAULTS = {"repetition_penalty": 1.1, "top_p": 0.8, "top_k": 20, "temperature": 0.7}


def make_handler(defaults):
    tm = MagicMock()
    tm.model_config.get_default_sampling_params.return_value = defaults
    tm.server_args = None  # not a ServerArgs: skips the metrics-label branch of the base class
    return OpenAIServingCompletion(tm, MagicMock())


def build(handler, **fields):
    req = CompletionRequest(model="m", prompt="hi", **fields)
    return handler._build_sampling_params(req)


def test_model_defaults_apply_when_request_sets_only_temperature():
    sp = build(make_handler(MODEL_DEFAULTS), temperature=0)
    assert sp["repetition_penalty"] == 1.1, sp["repetition_penalty"]
    assert sp["top_p"] == 0.8, sp["top_p"]
    assert sp["top_k"] == 20, sp["top_k"]
    assert sp["temperature"] == 0


def test_request_value_wins_over_model_default():
    sp = build(make_handler(MODEL_DEFAULTS), temperature=0, top_p=0.5)
    assert sp["top_p"] == 0.5


def test_schema_defaults_when_model_has_none():
    sp = build(make_handler({}), temperature=0)
    assert sp["repetition_penalty"] == 1.0 and sp["top_p"] == 1.0 and sp["top_k"] == -1


if __name__ == "__main__":
    import traceback
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn(); print("PASS", name)
            except Exception as e:  # noqa: BLE001
                failed += 1; print("FAIL", name, "->", type(e).__name__, str(e)[:200])
    raise SystemExit(1 if failed else 0)
