#!/usr/bin/env python3
"""3a probe (plan: PLAN.md). Same teacher-forced calls and output layout as followup/routing/scripts/engine_probe.py (so
analyze_engine.py reads it), plus the cached-token count of every call. Modes:
  mixed    all 320 calls (16 prompts x 4 served adapters x 5 candidates), 16 in flight, fixed shuffle seed 7
  smoke    the first 12 calls of that shuffle, serial (does a probe call report cached tokens at all?)
  control  cross-adapter control: (a0,p0) (a0,p1) (a1,p1) (a1,p2) on prompts sharing prefix A, run both as probe calls and
           as plain calls (max_tokens 1, no logprobs), serial; records cached tokens of each
  complete greedy 32-token completions of the 16 prompts for each adapter (the plan's fallback), text and token ids
Output JSON: {"engine","mode","stats","calls":[{"p","a","c","logp","cached"}]} (control/complete have their own records)."""
import argparse, asyncio, json, random, sys, time
import aiohttp

ap = argparse.ArgumentParser()
ap.add_argument("--engine", required=True, choices=["vllm", "sglang"])
ap.add_argument("--url", required=True)
ap.add_argument("--refs", required=True)
ap.add_argument("--mode", required=True, choices=["mixed", "smoke", "control", "complete"])
ap.add_argument("--concurrency", type=int, default=16)
ap.add_argument("--out", required=True)
a = ap.parse_args()
R = json.load(open(a.refs)); C = R["candidates"]; N = R["n_tokens"]; ADP = C[1:]
stats = {"errors": 0}

async def probe(s, name, ids, n_prompt, with_logprobs=True, max_new=1):
    """returns (logp list or None, cached_tokens or None, text)"""
    if a.engine == "vllm":
        body = {"model": name, "prompt": ids, "max_tokens": max_new, "temperature": 0}
        if with_logprobs: body["prompt_logprobs"] = 1
        async with s.post(a.url + "/v1/completions", json=body) as r:
            d = await r.json()
        det = (d.get("usage") or {}).get("prompt_tokens_details") or {}
        cached = det.get("cached_tokens")
        lp = None
        if with_logprobs:
            pl = d["choices"][0]["prompt_logprobs"]
            lp = [pl[i][str(ids[i])]["logprob"] for i in range(len(ids) - N, len(ids))]
        return lp, cached, d["choices"][0].get("text")
    body = {"input_ids": ids, "sampling_params": {"max_new_tokens": max_new, "temperature": 0}, "lora_path": name}
    if with_logprobs: body.update({"return_logprob": True, "logprob_start_len": n_prompt - 1})
    async with s.post(a.url + "/generate", json=body) as r:
        d = await r.json()
    mi = d["meta_info"]; cached = mi.get("cached_tokens"); lp = None
    if with_logprobs:
        e = mi["input_token_logprobs"]
        assert e[0][0] is None and len(e) == N + 1 and [x[1] for x in e][1:] == ids[-N:], "unexpected layout"
        lp = [x[0] for x in e][1:]
        assert all(x is not None for x in lp)
    return lp, cached, d.get("text")

async def one(s, p, ai, ci):
    ids = R["prompt_ids"][p] + R["tokens"][p][ci]; n_prompt = len(R["prompt_ids"][p])
    try:
        lp, cached, _ = await probe(s, ADP[ai], ids, n_prompt)
    except Exception as e:
        stats["errors"] += 1; print("error", p, ai, ci, repr(e)[:150], file=sys.stderr); lp = cached = None
    return {"p": p, "a": ai, "c": ci, "logp": lp, "cached": cached}

async def main():
    out = {"engine": a.engine, "mode": a.mode, "refs": a.refs}
    t0 = time.time()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=900), connector=aiohttp.TCPConnector(limit=0)) as s:
        if a.mode in ("mixed", "smoke"):
            calls = [(p, ai, ci) for p in range(len(R["prompts"])) for ai in range(4) for ci in range(5)]
            random.Random(7).shuffle(calls)
            if a.mode == "smoke": calls = calls[:12]
            sem = asyncio.Semaphore(a.concurrency if a.mode == "mixed" else 1)
            async def g(c):
                async with sem: return await one(s, *c)
            out["calls"] = await asyncio.gather(*[g(c) for c in calls])
        elif a.mode == "control":
            rec = []
            for kind in ("probe", "plain"):
                for (ai, p) in [(0, 0), (0, 1), (1, 1), (1, 2)]:
                    ids = R["prompt_ids"][p] + R["tokens"][p][1 + ai]; n_prompt = len(R["prompt_ids"][p])
                    pid = R["prompt_ids"][p] if kind == "plain" else ids
                    lp, cached, _ = await probe(s, ADP[ai], pid, n_prompt, with_logprobs=(kind == "probe"))
                    rec.append({"kind": kind, "adapter": ai, "prompt": p, "prompt_tokens": len(pid), "cached": cached})
            out["control"] = rec
        else:
            rec = []
            for ai in range(4):
                for p in range(len(R["prompts"])):
                    ids = R["prompt_ids"][p]
                    _, cached, text = await probe(s, ADP[ai], ids, len(ids), with_logprobs=False, max_new=32)
                    rec.append({"a": ai, "p": p, "text": text, "cached": cached})
            out["complete"] = rec
    out["stats"] = stats; out["seconds"] = time.time() - t0
    json.dump(out, open(a.out, "w"))
    print(f"{a.engine} {a.mode}: done in {out['seconds']:.0f}s; stats {stats}")
asyncio.run(main())
