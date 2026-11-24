#!/usr/bin/env python3
"""Item C (Amendment 5): B8 server-level reproduction. Usage: item_c_repro.py PROMPTS_JSONL MODEL_DIR OUT_JSON"""
import json, sys, requests
from transformers import AutoTokenizer
P, M, OUT = sys.argv[1:4]
tok = AutoTokenizer.from_pretrained(M)
rows = [json.loads(l) for l in open(P)]
GEN = dict(repetition_penalty=1.1, top_p=0.8, top_k=20)
S = requests.Session(); B = "http://localhost:30000"
def call(ep, r, extra):
    text = tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True)
    if ep == "chat":
        j = S.post(B + "/v1/chat/completions", json=dict(model="qwen", messages=r["messages"], temperature=0, max_tokens=128, **extra), timeout=300).json()
        return j["choices"][0]["message"]["content"]
    if ep == "completions":
        j = S.post(B + "/v1/completions", json=dict(model="qwen", prompt=text, temperature=0, max_tokens=128, **extra), timeout=300).json()
        return j["choices"][0]["text"]
    sp = dict(temperature=0, max_new_tokens=128, **extra)
    return S.post(B + "/generate", json=dict(text=text, sampling_params=sp), timeout=300).json()["text"]
res = {}
for ep in ("chat", "completions", "generate"):
    a = [call(ep, r, {}) for r in rows]; b = [call(ep, r, GEN) for r in rows]; c = [call(ep, r, {}) for r in rows]
    res[ep] = dict(n=len(rows), i_vs_ii=sum(x != y for x, y in zip(a, b)), i_vs_repeat=sum(x != y for x, y in zip(a, c)),
                   texts=dict(i=a, ii=b, repeat=c))
    print(ep, {k: v for k, v in res[ep].items() if k != "texts"}, flush=True)
json.dump(res, open(OUT, "w"))
