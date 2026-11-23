import asyncio, aiohttp, json, sys, random
sys.path.insert(0, "/root/k/scripts")
import numpy as np
from transformers import AutoTokenizer
import loadgen
M = "/root/k/model"; TRIALS = 12; TARGETS = [32, 106, 2, 20, 136, 5, 50]
tok = AutoTokenizer.from_pretrained(M)
total = 170
rng = np.random.default_rng(5)
prompts = loadgen.make_prompts(tok, total, 5, 256, 0)
names = [f"adapter_{i:04d}" for i in range(8)]
picks = [names[i] for i in rng.choice(8, size=total, p=np.full(8, 1 / 8))]
async def one(s, prompt, adapter):
    body = {"model": "default", "prompt": prompt, "max_tokens": 1, "temperature": 0, "adapter": adapter, "logprobs": 5}
    async with s.post("http://localhost:1234/v1/completions", json=body) as r:
        d = await r.json()
    c = d["choices"][0]; lp = None
    try:
        lp = c["logprobs"]["content"][0]
        top = [(t.get("token"), round(t["logprob"], 3)) for t in lp["top_logprobs"]]
    except Exception:
        top = None
    return {"text": c.get("text"), "top": top}
async def main():
    out = {}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as s:
        for t in TARGETS:
            ad = picks[t]; res = []
            for k in range(TRIALS):
                r = random.Random(1000 * t + k)
                others = [i for i in range(total) if i != t and picks[i] != ad]
                comp = []; used = {ad}
                for i in r.sample(others, len(others)):
                    if picks[i] not in used: used.add(picks[i]); comp.append(i)
                    if len(comp) == 7: break
                reqs = [one(s, prompts[t], ad)] + [one(s, prompts[i], picks[i]) for i in comp]
                got = await asyncio.gather(*reqs)
                res.append(got[0])
            out[str(t)] = {"adapter": ad, "trials": res}
    json.dump(out, open(sys.argv[1], "w"))
asyncio.run(main())
