#!/usr/bin/env python3
"""Teacher-forced scoring of an engine against the reference tokens. Stdlib + transformers (tokenizer only, for engines that take text).
usage: tf_client.py ENGINE PORT REF_JSONL OUT [--conc N] [--limit N] [--probe] [--tokdir DIR] [--rep K]
Writes one JSON line per prompt: id, L (prompt len), G (gen len), lp_engine (list, logprob of each reference gen token or None), method, usage, cached_tokens, error."""
import json, sys, time, argparse, threading, urllib.request
ap = argparse.ArgumentParser(); ap.add_argument("engine"); ap.add_argument("port"); ap.add_argument("ref"); ap.add_argument("out")
ap.add_argument("--conc", type=int, default=1); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--probe", action="store_true")
ap.add_argument("--tokdir"); ap.add_argument("--model", default=None); ap.add_argument("--sampling", choices=["neutral", "defaults"], default="neutral", help="neutral: explicit repetition_penalty=1, top_k off, top_p=1 on generation requests; defaults: send only temperature=0 (what a user gets)"); ap.add_argument("--gen", action="store_true", help="let the engine generate (needed for cache cells); score generated tokens up to the first divergence")
a = ap.parse_args(); base = f"http://localhost:{a.port}"
MODEL = a.model or ("default" if a.engine == "mistralrs" else "m")
tok = None
if a.tokdir:
    from transformers import AutoTokenizer; tok = AutoTokenizer.from_pretrained(a.tokdir)
NEUT = ({"vllm": {"repetition_penalty": 1.0, "top_k": -1, "top_p": 1.0}, "sglang_sp": {"repetition_penalty": 1.0, "top_k": -1, "top_p": 1.0}, "mistralrs": {"repetition_penalty": 1.0, "top_k": 0, "top_p": 1.0}} if a.sampling == "neutral" else {"vllm": {}, "sglang_sp": {}, "mistralrs": {}})
def post(path, body, timeout=900):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r: return json.loads(r.read())
    except urllib.error.HTTPError as e: return {"_error": f"HTTP {e.code}", "_body": e.read().decode()[:400]}
    except Exception as e: return {"_error": str(e)}
def score(rec):
    L, G = len(rec["prompt_ids"]), len(rec["gen_ids"]); full = rec["prompt_ids"] + rec["gen_ids"]
    out = {"id": rec["id"], "L": L, "G": G, "lp_engine": None, "method": None, "usage": None, "cached_tokens": None, "error": None}
    if a.gen and a.engine != "mistralrs":   # generation path for vllm/sglang (mistral.rs always uses it)
        out["method"] = f"{a.engine} generate (prefix until divergence)"
        if a.engine == "vllm":
            d = post("/v1/completions", {"model": MODEL, "prompt": rec["prompt_ids"], "max_tokens": G, "temperature": 0, "logprobs": 1, "return_tokens_as_token_ids": True, **NEUT["vllm"]})
            if "_error" in d: out["error"] = d; return out
            ch = d["choices"][0]; lg = ch.get("logprobs") or {}; out["usage"] = d.get("usage"); out["cached_tokens"] = ((d.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
            out["gen_ids_engine"] = [int(t.split(":")[1]) for t in lg.get("tokens", [])]; out["lp_gen_engine"] = lg.get("token_logprobs"); out["finish"] = ch.get("finish_reason")
        else:
            d = post("/generate", {"input_ids": rec["prompt_ids"], "sampling_params": {"max_new_tokens": G, "temperature": 0, **NEUT["sglang_sp"]}, "return_logprob": True})
            if isinstance(d, list): d = d[0]
            if "_error" in d: out["error"] = d; return out
            mi = d.get("meta_info", {}); otl = mi.get("output_token_logprobs") or []; out["cached_tokens"] = mi.get("cached_tokens"); out["usage"] = {"prompt_tokens": mi.get("prompt_tokens"), "completion_tokens": mi.get("completion_tokens")}
            out["gen_ids_engine"] = [x[1] for x in otl]; out["lp_gen_engine"] = [x[0] for x in otl]; out["finish"] = (mi.get("finish_reason") or {}).get("type")
        return out
    if a.engine == "vllm":
        d = post("/v1/completions", {"model": MODEL, "prompt": full, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 1}); out["method"] = "vllm prompt_logprobs"
        if "_error" in d: out["error"] = d; return out
        pl = d["choices"][0].get("prompt_logprobs"); out["usage"] = d.get("usage"); out["cached_tokens"] = ((d.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
        if pl: out["lp_engine"] = [(pl[i].get(str(full[i])) or {}).get("logprob") if pl[i] else None for i in range(L, L + G)]
        out["_raw_head"] = json.dumps(pl[L:L + 2])[:300] if pl else None
    elif a.engine == "sglang":
        d = post("/generate", {"input_ids": full, "sampling_params": {"max_new_tokens": 1, "temperature": 0}, "return_logprob": True, "logprob_start_len": L - 1, "top_logprobs_num": 1}); out["method"] = "sglang return_logprob"
        if isinstance(d, list): d = d[0]
        if "_error" in d: out["error"] = d; return out
        mi = d.get("meta_info", {}); itl = mi.get("input_token_logprobs"); out["cached_tokens"] = mi.get("cached_tokens"); out["usage"] = {"prompt_tokens": mi.get("prompt_tokens"), "completion_tokens": mi.get("completion_tokens")}
        if itl: out["lp_engine"] = [x[0] for x in itl][-G:] if len(itl) >= G else None; out["_n_returned"] = len(itl); out["_ids_ok"] = [x[1] for x in itl][-G:] == rec["gen_ids"]  # start at L-1 so the first generated token also gets a logprob (entry 0 is token L-1 and is null)
        out["_raw_head"] = json.dumps(itl[:2])[:300] if itl else None
    elif a.engine == "mistralrs":   # no prompt logprobs on this engine: generate greedily and compare generated-token logprobs until the first divergence
        text = tok.decode(rec["prompt_ids"], skip_special_tokens=False)
        d = post("/v1/completions", {"model": MODEL, "prompt": text, "max_tokens": G, "temperature": 0, "logprobs": 1, **NEUT["mistralrs"]}); out["method"] = "mistralrs generate (prefix until divergence)"
        if "_error" in d: out["error"] = d; return out
        out["usage"] = d.get("usage"); ch = d["choices"][0]; cont = (ch.get("logprobs") or {}).get("content") or []
        # known layout quirk (BUGS B3): top_logprobs[].token carries the token ID and the text; for greedy decoding top-1 is the chosen token
        out["gen_ids_engine"] = [int(c["top_logprobs"][0]["token"]) if c.get("top_logprobs") else None for c in cont]
        out["lp_gen_engine"] = [c.get("logprob") for c in cont]; out["finish"] = ch.get("finish_reason"); out["prompt_tokens_match"] = (out["usage"] or {}).get("prompt_tokens") == L
    return out
recs = [json.loads(l) for l in open(a.ref)]
if a.limit: recs = recs[:a.limit]
if a.probe:
    for r in recs[:2]:
        o = score(r); print(json.dumps({k: v for k, v in o.items() if k != "lp_engine"}, default=str)[:900]); print("lp_engine[:5]:", (o["lp_engine"] or [])[:5], "| ref lp[:5]:", [round(x, 4) for x in r["lp"][:5]])
    sys.exit()
res = {}; lock = threading.Lock(); it = iter(range(len(recs)))
def w():
    while True:
        with lock:
            try: i = next(it)
            except StopIteration: return
        o = score(recs[i])
        with lock: res[i] = o
ts = [threading.Thread(target=w) for _ in range(a.conc)]; t0 = time.time(); [t.start() for t in ts]; [t.join() for t in ts]
with open(a.out, "w") as f:
    for i in range(len(recs)): f.write(json.dumps(res[i], default=str) + "\n")
print("scored", len(recs), "in", round(time.time() - t0, 1), "s; errors", sum(1 for o in res.values() if o["error"]), "; missing lp", sum(1 for o in res.values() if o["lp_engine"] is None))
