#!/usr/bin/env python3
"""Open-loop load generator for the multi-LoRA study.

Requests arrive as a Poisson process at a fixed rate, independent of how fast the
engine answers. Each request streams, so time to first token (TTFT) and time per
output token (TPOT) are measured per request. The first --warmup requests are sent
but excluded from the summary. Adapter popularity is uniform or Zipf.
"""
import argparse
import asyncio
import json
import math
import random
import time

import aiohttp
import numpy as np
from transformers import AutoTokenizer

REQUEST_TIMEOUT_S = 180


def make_prompts(tok, n: int, seed: int, tokens: int) -> list[str]:
    rng = random.Random(seed)
    words = []
    for i in range(tok.vocab_size):
        s = tok.decode([i])
        if s.startswith(" ") and 3 <= len(s) <= 9 and s[1:].isalpha() and s[1:].isascii():
            words.append(s)
    prompts = []
    for _ in range(n):
        text = "".join(rng.choice(words) for _ in range(tokens + 16))
        ids = tok(text).input_ids[:tokens]
        prompts.append(tok.decode(ids))
    return prompts


def zipf_probs(n: int, s: float = 1.0) -> np.ndarray:
    w = 1.0 / np.arange(1, n + 1) ** s
    return w / w.sum()


def build_body(engine, adapter, prompt, base_model, max_tokens):
    body = {"prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "stream": True}
    if engine in ("vllm", "sglang"):
        body["ignore_eos"] = True
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
    return body


async def one_request(session, url, body, t_due, t0, rec):
    delay = t_due - (time.perf_counter() - t0)
    if delay > 0:
        await asyncio.sleep(delay)
    t_send = time.perf_counter()
    rec["send_lag_s"] = (t_send - t0) - t_due
    n_chunks = 0
    t_first = None
    t_last = None
    try:
        async with session.post(url + "/v1/completions", json=body,
                                timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S)) as resp:
            if resp.status != 200:
                rec["error"] = f"http {resp.status}"
                return
            async for raw in resp.content:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    d = json.loads(data)
                except ValueError:
                    continue
                ch = d.get("choices") or []
                if ch and ch[0].get("text"):
                    now = time.perf_counter()
                    if t_first is None:
                        t_first = now
                    t_last = now
                    n_chunks += 1
    except Exception as e:  # noqa: BLE001
        rec["error"] = type(e).__name__
        return
    if t_first is None:
        rec["error"] = "no tokens"
        return
    rec["ttft_s"] = t_first - t_send
    rec["e2e_s"] = t_last - t_send
    rec["output_chunks"] = n_chunks
    rec["tpot_s"] = (t_last - t_first) / (n_chunks - 1) if n_chunks > 1 else None


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else None


def summarize(recs, warmup, duration_s):
    meas = recs[warmup:]
    ok = [r for r in meas if "error" not in r]
    ttft = [r["ttft_s"] * 1000 for r in ok]
    tpot = [r["tpot_s"] * 1000 for r in ok if r.get("tpot_s") is not None]
    e2e = [r["e2e_s"] * 1000 for r in ok]
    toks = sum(r["output_chunks"] for r in ok)
    return {
        "measured": len(meas), "ok": len(ok), "errors": len(meas) - len(ok),
        "ttft_ms": {"p50": pct(ttft, 50), "p95": pct(ttft, 95), "p99": pct(ttft, 99)},
        "tpot_ms": {"p50": pct(tpot, 50), "p95": pct(tpot, 95)},
        "e2e_ms": {"p50": pct(e2e, 50), "p95": pct(e2e, 95)},
        "output_tokens_per_s": toks / duration_s if duration_s > 0 else None,
        "requests_per_s": len(ok) / duration_s if duration_s > 0 else None,
        "max_send_lag_s": max((r["send_lag_s"] for r in meas), default=None),
    }


async def run(args):
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    total = args.warmup + args.requests
    rng = np.random.default_rng(args.seed)
    gaps = rng.exponential(1.0 / args.rate, size=total)
    due = np.cumsum(gaps)
    prompts = make_prompts(tok, total, args.seed, args.input_tokens)
    names = [f"adapter_{args.adapter_offset + i:04d}" for i in range(args.adapters)]
    if args.adapters == 0:
        picks = [None] * total
    else:
        p = zipf_probs(args.adapters) if args.popularity == "zipf" else np.full(args.adapters, 1 / args.adapters)
        picks = [names[i] for i in rng.choice(args.adapters, size=total, p=p)]
    recs = [{"i": i, "adapter": picks[i]} for i in range(total)]
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=conn) as session:
        t0 = time.perf_counter()
        tasks = [asyncio.create_task(one_request(
            session, args.url, build_body(args.engine, picks[i], prompts[i], args.base_model_name,
                                          args.output_tokens), float(due[i]), t0, recs[i]))
            for i in range(total)]
        await asyncio.gather(*tasks)
        t_end = time.perf_counter()
    # measured window: from the first measured request's due time to the last completion
    window = (t_end - t0) - float(due[args.warmup - 1] if args.warmup else 0.0)
    out = {
        "config": {k: v for k, v in vars(args).items()},
        "summary": summarize(recs, args.warmup, window),
        "requests": recs,
    }
    json.dump(out, open(args.out, "w"))
    s = out["summary"]
    print(json.dumps({"rate": args.rate, "ok": s["ok"], "errors": s["errors"],
                      "ttft_ms": s["ttft_ms"], "tpot_ms": s["tpot_ms"],
                      "tok_per_s": round(s["output_tokens_per_s"] or 0, 1),
                      "max_send_lag_s": round(s["max_send_lag_s"] or 0, 3)}))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", required=True, choices=["vllm", "sglang", "mistralrs"])
    ap.add_argument("--url", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--base-model-name", default="qwen")
    ap.add_argument("--rate", type=float, required=True, help="requests per second")
    ap.add_argument("--requests", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--adapters", type=int, default=0, help="number of adapters; 0 = base model only")
    ap.add_argument("--adapter-offset", type=int, default=0)
    ap.add_argument("--popularity", choices=["uniform", "zipf"], default="zipf")
    ap.add_argument("--input-tokens", type=int, default=256)
    ap.add_argument("--output-tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
