#!/usr/bin/env python3
"""C8 by endpoint: with ONLY temperature=0 in the request, does the output equal the raw-greedy reference, the HF-default (penalized) reference, or neither?
usage: c8_endpoints.py ENGINE PORT MODEL_DIR RAW_REF HFDEF_REF OUT   (stdlib + transformers tokenizer). Endpoints: chat (messages), completions (token-id list; text for mistralrs).
Classification per prompt compares the generated TEXT to the decoded reference text (prefix-equal up to the shorter length, minimum 20 chars)."""
import json, sys, urllib.request
from transformers import AutoTokenizer
E, port, mdir, rawp, hfp, outp = sys.argv[1:7]; base = f"http://localhost:{port}"; MODEL = "default" if E == "mistralrs" else "m"
tok = AutoTokenizer.from_pretrained(mdir); raw = [json.loads(l) for l in open(rawp)]; hf = [json.loads(l) for l in open(hfp)]
prompts = [json.loads(l) for l in open("/root/kvc/diff/prompts/prompts_main.jsonl")]
def post(path, body):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r: return json.loads(r.read())
    except urllib.error.HTTPError as e: return {"_error": f"HTTP {e.code}", "_body": e.read().decode()[:300]}
    except Exception as e: return {"_error": str(e)}
def same(a, b):
    n = min(len(a), len(b)); return n >= 20 and a[:n] == b[:n]
res = {}
for ep in ("chat", "completions"):
    cls = {"raw": 0, "hfdefault": 0, "both": 0, "neither": 0, "error": 0}; rows = []
    for i, p in enumerate(prompts):
        if ep == "chat": d = post("/v1/chat/completions", {"model": MODEL, "messages": p["messages"], "max_tokens": 128, "temperature": 0}); txt = (d.get("choices") or [{}])[0].get("message", {}).get("content")
        else:
            prompt = raw[i]["prompt_ids"] if E != "mistralrs" else tok.decode(raw[i]["prompt_ids"], skip_special_tokens=False)
            d = post("/v1/completions", {"model": MODEL, "prompt": prompt, "max_tokens": 128, "temperature": 0}); txt = (d.get("choices") or [{}])[0].get("text")
        if txt is None: cls["error"] += 1; rows.append({"id": i, "err": d.get("_error") or d.get("_body")}); continue
        tr = tok.decode(raw[i]["gen_ids"], skip_special_tokens=True); th = tok.decode(hf[i]["gen_ids"], skip_special_tokens=True)
        a, b = same(txt, tr), same(txt, th); cls["both" if a and b else "raw" if a else "hfdefault" if b else "neither"] += 1; rows.append({"id": i, "raw": a, "hf": b})
    res[ep] = cls; print(E, ep, cls, flush=True)
json.dump({"engine": E, "result": res}, open(outp, "w"), indent=1)
