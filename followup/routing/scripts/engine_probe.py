#!/usr/bin/env python3
"""A2 probe: teacher-forced log-probabilities from a running engine, for every (prompt, served adapter, candidate) call.

For each prompt p, served adapter a (adapters 0 to 3) and candidate c (base and adapters 0 to 3): the engine is given
prompt(p) + T[p][c] (the fp32 reference continuation of candidate c) with adapter a selected, and the log-probability of
each of the 128 forced tokens is recorded. Plan: ../PLAN.md. Output JSON: {"calls": [{"p","a","c","logp":[128 floats|null]}], "stats": {...}}.
Engines: vllm (prompt_logprobs), sglang (input_token_logprobs), mistralrs (one request per position; top-list lookup)."""
import argparse, asyncio, json, random, sys, time
import aiohttp

ap = argparse.ArgumentParser()
ap.add_argument("--engine", required=True, choices=["vllm", "sglang", "mistralrs"])
ap.add_argument("--url", required=True)
ap.add_argument("--refs", required=True, help="refs_s*.json from refs.py")
ap.add_argument("--mode", required=True, choices=["serial", "mixed"])
ap.add_argument("--concurrency", type=int, default=16)
ap.add_argument("--tokenizer", help="needed for mistralrs (decode prefixes to text)")
ap.add_argument("--topn", type=int, default=20, help="mistralrs: logprobs count requested")
ap.add_argument("--limit", type=int, default=0, help="smoke tests: only the first N calls")
ap.add_argument("--out", required=True)
a = ap.parse_args()
R = json.load(open(a.refs)); C = R["candidates"]; N = R["n_tokens"]; ADP = C[1:]
tok = None
if a.engine == "mistralrs":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
stats = {"positions": 0, "missing_topn": 0, "dropped_roundtrip": 0, "errors": 0}

async def call_vllm(s, name, ids):
    body = {"model": name, "prompt": ids, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 1}
    async with s.post(a.url + "/v1/completions", json=body) as r:
        d = await r.json()
    pl = d["choices"][0]["prompt_logprobs"]
    return [pl[i][str(ids[i])]["logprob"] for i in range(len(ids) - N, len(ids))]

async def call_sglang(s, name, ids, n_prompt):
    body = {"input_ids": ids, "sampling_params": {"max_new_tokens": 1, "temperature": 0}, "return_logprob": True,
            "logprob_start_len": n_prompt - 1, "lora_path": name}
    async with s.post(a.url + "/generate", json=body) as r:
        d = await r.json()
    lp = d["meta_info"]["input_token_logprobs"]            # entries [logprob, token_id, text|None] from logprob_start_len on; the first has no logprob
    assert lp[0][0] is None and len(lp) == N + 1, f"unexpected layout: {len(lp)} entries, first {lp[0][0]}"
    out = [e[0] for e in lp][1:]
    got = [e[1] for e in lp][1:]
    assert got == ids[-N:], "returned token ids do not match the forced ids"
    assert all(x is not None for x in out), "None logprob in the forced range"
    return out

async def call_mistralrs(s, name, ids, n_prompt):
    out = []
    for t in range(N):
        prefix = ids[: n_prompt + t]
        text = tok.decode(prefix)
        stats["positions"] += 1
        if tok(text).input_ids != prefix:                    # decode/re-encode did not give back the same ids
            stats["dropped_roundtrip"] += 1; out.append(None); continue
        body = {"model": "default", "prompt": text, "max_tokens": 1, "temperature": 0, "logprobs": a.topn, "adapter": name}
        async with s.post(a.url + "/v1/completions", json=body) as r:
            d = await r.json()
        top = d["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        want = ids[n_prompt + t]
        hit = [e["logprob"] for e in top if e.get("token") == want]    # v0.9.5 layout: token id in "token"
        if hit: out.append(hit[0])
        else:
            stats["missing_topn"] += 1; out.append(min(e["logprob"] for e in top))   # stated weakness: floor value
    return out

async def one(s, p, ai, ci):
    name = ADP[ai]; ids = R["prompt_ids"][p] + R["tokens"][p][ci]; n_prompt = len(R["prompt_ids"][p])
    try:
        if a.engine == "vllm": lp = await call_vllm(s, name, ids)
        elif a.engine == "sglang": lp = await call_sglang(s, name, ids, n_prompt)
        else: lp = await call_mistralrs(s, name, ids, n_prompt)
    except Exception as e:
        stats["errors"] += 1; print("error", p, ai, ci, repr(e)[:150], file=sys.stderr); lp = None
    return {"p": p, "a": ai, "c": ci, "logp": lp}

async def main():
    calls = [(p, ai, ci) for p in range(len(R["prompts"])) for ai in range(4) for ci in range(5)]
    random.Random(7).shuffle(calls)                           # fixed shuffle: the mixed mode keeps all adapters in flight together
    if a.limit: calls = calls[: a.limit]
    sem = asyncio.Semaphore(1 if a.mode == "serial" else a.concurrency)
    t0 = time.time(); res = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=900), connector=aiohttp.TCPConnector(limit=0)) as s:
        async def guarded(c):
            async with sem: return await one(s, *c)
        res = await asyncio.gather(*[guarded(c) for c in calls])
    json.dump({"engine": a.engine, "mode": a.mode, "refs": a.refs, "stats": stats, "seconds": time.time() - t0, "calls": res}, open(a.out, "w"))
    print(f"{a.engine} {a.mode}: {len(res)} calls in {time.time()-t0:.0f}s; stats {stats}")
asyncio.run(main())
