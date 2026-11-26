#!/usr/bin/env python3
"""Step 9 correctness check: send N concurrent greedy requests (fixed prompts, adapters round-robin) and save the
completions, so two servers' outputs can be compared token for token. Usage:
  greedy_check.py run URL OUT.json [N]      (default N = 96, max_tokens 64, temperature 0)
  greedy_check.py diff A.json B.json"""
import asyncio, json, sys

import aiohttp

PROMPT = "Write a short paragraph about the history of the number {i}, then list three facts about it."


async def one(session, url, i):
    body = {"model": f"adapter_{i % 32:04d}", "prompt": PROMPT.format(i=i), "max_tokens": 64, "temperature": 0}
    async with session.post(f"{url}/v1/completions", json=body) as r:
        j = await r.json()
        return i, j["choices"][0]["text"]


async def run(url, out, n):
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as s:
        res = await asyncio.gather(*[one(s, url, i) for i in range(n)])
    json.dump({str(i): t for i, t in res}, open(out, "w"), indent=0)
    print(f"saved {len(res)} completions to {out}")


def diff(a, b):
    A, B = json.load(open(a)), json.load(open(b))
    same = sum(A[k] == B[k] for k in A)
    first = []
    for k in A:
        if A[k] != B[k]:
            n = next((i for i, (x, y) in enumerate(zip(A[k], B[k])) if x != y), min(len(A[k]), len(B[k])))
            first.append(n)
    print(f"{a} vs {b}: identical {same} of {len(A)}; differing requests diverge at character "
          f"{sorted(first)[:10]}{' ...' if len(first) > 10 else ''}")


if __name__ == "__main__":
    if sys.argv[1] == "run": asyncio.run(run(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 96))
    else: diff(sys.argv[2], sys.argv[3])
