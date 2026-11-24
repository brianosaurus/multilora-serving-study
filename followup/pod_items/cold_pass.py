#!/usr/bin/env python3
"""Amendment 7: usage cold_pass.py OUT_JSON PROMPTS_JSONL. Two back-to-back passes of the 64 prompts on /v1/chat/completions, temperature 0 only, 128 tokens, one request at a time."""
import json, sys, requests
rows = [json.loads(l) for l in open(sys.argv[2])]
S = requests.Session()
def ask(r):
    j = S.post("http://localhost:30000/v1/chat/completions", json=dict(model="qwen", messages=r["messages"], temperature=0, max_tokens=128), timeout=300).json()
    return j["choices"][0]["message"]["content"]
p1 = [ask(r) for r in rows]; p2 = [ask(r) for r in rows]
diff = [k for k, (a, b) in enumerate(zip(p1, p2)) if a != b]
json.dump(dict(pass1=p1, pass2=p2, differing=diff), open(sys.argv[1], "w"))
print("pass1 vs pass2 differ on", len(diff), "of", len(rows), "prompts:", diff)
