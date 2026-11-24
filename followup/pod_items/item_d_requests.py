#!/usr/bin/env python3
"""Item D (Amendment 6): usage item_d_requests.py OUT_JSON  (server on localhost:1234)"""
import json, sys, requests
S = requests.Session(); P = "The capital of France is"
def ask(adapter):
    b = dict(model="default", prompt=P, temperature=0, max_tokens=128)
    if adapter: b["adapter"] = adapter
    j = S.post("http://localhost:1234/v1/completions", json=b, timeout=600).json()
    try: return j["choices"][0]["text"]
    except Exception: return "ERR " + json.dumps(j)[:300]
names = [f"adapter_{i:04d}" for i in range(8)]
out = {"base": [ask(None) for _ in range(3)]}
for n in names: out[n] = [ask(n) for _ in range(6)]
for n in names: out[n + "_round2"] = [ask(n) for _ in range(3)]
json.dump(out, open(sys.argv[1], "w"), indent=1)
