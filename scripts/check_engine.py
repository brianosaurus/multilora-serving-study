#!/usr/bin/env python3
"""Adapter identity check for one running engine, against the PEFT reference.

Implements the criteria in docs/PLAN.md. For each (adapter, prompt) request the engine's
text is compared by longest common prefix (in characters) with the reference text of the
base model and of every adapter. The request passes only if the longest shared prefix is
with the correct adapter's reference; a tie is a failure. Two modes: serial (one request
at a time) and mixed (all adapters' requests sent concurrently).
"""
import argparse
import json
import random
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from transformers import AutoTokenizer

PASS_THRESHOLD = 0.90
MAX_TOKENS = 32


def build_request(engine: str, adapter: str | None, prompt: str, base_model: str) -> dict:
    body = {"prompt": prompt, "max_tokens": MAX_TOKENS, "temperature": 0}
    if engine == "vllm":
        body["model"] = adapter or base_model
    elif engine == "sglang":
        body["model"] = base_model
        if adapter:
            body["lora_path"] = adapter
    elif engine == "mistralrs":
        body["model"] = "default"
        if adapter:
            body["adapter"] = adapter
    else:
        raise ValueError(engine)
    return body


def post(url: str, body: dict, timeout: int = 180) -> str:
    req = urllib.request.Request(
        url + "/v1/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return d["choices"][0]["text"]


def lcp(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def score(text: str, adapter: str, prompt_idx: int, ref_text: dict) -> dict:
    scores = {v: lcp(text, ref_text[v][prompt_idx]) for v in ref_text}
    best_other = max(s for v, s in scores.items() if v != adapter)
    return {"scores": scores, "correct": scores[adapter] > best_other}


def run_mode(name, tasks, fn, ref_text):
    results = []
    t0 = time.time()
    outs = fn(tasks)
    for (adapter, i), text in zip(tasks, outs):
        r = score(text, adapter, i, ref_text)
        r.update({"adapter": adapter, "prompt": i, "text": text})
        results.append(r)
    per_adapter = {}
    for r in results:
        per_adapter.setdefault(r["adapter"], []).append(r["correct"])
    summary = {
        "mode": name,
        "requests": len(results),
        "correct": sum(r["correct"] for r in results),
        "accuracy": sum(r["correct"] for r in results) / len(results),
        "per_adapter": {a: sum(v) / len(v) for a, v in per_adapter.items()},
        "seconds": round(time.time() - t0, 1),
    }
    summary["pass"] = summary["accuracy"] >= PASS_THRESHOLD
    return summary, results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", required=True, choices=["vllm", "sglang", "mistralrs"])
    ap.add_argument("--url", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--tokenizer", required=True, help="base model directory")
    ap.add_argument("--base-model-name", default="qwen")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ref = json.load(open(args.reference))
    prompts = ref["prompts"]
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    ref_text = {v: [tok.decode(ids, skip_special_tokens=True) for ids in outs]
                for v, outs in ref["variants"].items()}
    adapters = [v for v in ref["variants"] if v != "base"]

    def call(task):
        adapter, i = task
        return post(args.url, build_request(args.engine, adapter, prompts[i], args.base_model_name))

    serial_tasks = [(a, i) for a in adapters for i in range(len(prompts))]
    mixed_tasks = list(serial_tasks)
    random.Random(0).shuffle(mixed_tasks)

    def serial(tasks):
        return [call(t) for t in tasks]

    def mixed(tasks):
        with ThreadPoolExecutor(max_workers=len(tasks)) as ex:
            return list(ex.map(call, tasks))

    base_sanity = post(args.url, build_request(args.engine, None, prompts[0], args.base_model_name))
    s_sum, s_res = run_mode("serial", serial_tasks, serial, ref_text)
    m_sum, m_res = run_mode("mixed", mixed_tasks, mixed, ref_text)

    report = {
        "engine": args.engine,
        "threshold": PASS_THRESHOLD,
        "adapters": adapters,
        "base_sanity_reply": base_sanity,
        "serial": s_sum,
        "mixed": m_sum,
        "overall_pass": s_sum["pass"] and m_sum["pass"],
        "details": {"serial": s_res, "mixed": m_res},
    }
    json.dump(report, open(args.out, "w"), indent=1)
    for s in (s_sum, m_sum):
        print(f"{args.engine} {s['mode']:<6} {s['correct']}/{s['requests']} = {s['accuracy']:.3f}  "
              f"pass={s['pass']}  per-adapter={ {a: round(v, 2) for a, v in s['per_adapter'].items()} }")
    print(f"{args.engine} OVERALL pass={report['overall_pass']}")


if __name__ == "__main__":
    main()
