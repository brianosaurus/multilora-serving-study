#!/usr/bin/env python3
"""Part A (A1, A2): where the bf16-vs-fp32 noise comes from, computed from refs_out.json (refs.py pipeline, B scale 0.05)."""
import json, sys
import numpy as np
d = json.load(open(sys.argv[1])); C = d["candidates"]; P = len(d["prompts"])
L32 = {v: np.array(d["L32"][v]) for v in C}; LBF = {v: np.array(d["Lbf"][v]) for v in C}   # [p][c][t]
def own(L, v):        # variant v on its own continuation: [p][t]
    return L[v][:, C.index(v), :]
pct = lambda a, q: float(np.percentile(a, q))
print("== A1: base model, no adapter, on the base model's own fp32 greedy continuations (16 prompts x 128 tokens)")
base_d = np.abs(own(LBF, "base") - own(L32, "base"))            # [p][t]
per_prompt = base_d.mean(axis=1)
print(f"per-prompt mean |bf16 - fp32| : median {np.median(per_prompt):.4f}  min {per_prompt.min():.4f}  max {per_prompt.max():.4f}  mean {per_prompt.mean():.4f}")
print("   per-prompt values:", [round(float(x), 3) for x in per_prompt])
print(f"per-token distribution (2048 positions): p50 {pct(base_d,50):.4f}  p90 {pct(base_d,90):.4f}  p99 {pct(base_d,99):.4f}  max {base_d.max():.3f}")
print("\n== for comparison, the adapters (64 request values = 16 prompts x 4 adapters)")
ad = {a: np.abs(own(LBF, a) - own(L32, a)) for a in C[1:]}      # [p][t]
req = np.array([ad[a][p].mean() for a in C[1:] for p in range(P)])
print(f"per-request mean: median {np.median(req):.4f} (the 'noise' used by the rule)  min {req.min():.4f}  max {req.max():.4f}")
print("   by adapter (median over prompts):", {a: round(float(np.median(ad[a].mean(axis=1))), 4) for a in C[1:]})
print("\n== A2: per-token absolute logprob differences, pooled over the 64 (prompt, adapter) requests = 8192 positions")
allpos = np.concatenate([ad[a].reshape(-1) for a in C[1:]])
print(f"percentiles: p50 {pct(allpos,50):.4f}  p90 {pct(allpos,90):.4f}  p99 {pct(allpos,99):.4f}  max {allpos.max():.3f}")
print(f"share of positions above 1 nat: {np.mean(allpos>1):.3%}   above 5 nats: {np.mean(allpos>5):.3%}")
s = np.sort(allpos)[::-1]; k = int(round(0.05 * len(s)))
print(f"mean over all positions {allpos.mean():.4f}; the top 5% of positions ({k}) contribute {s[:k].sum()/s.sum():.1%} of the sum; mean without them {s[k:].mean():.4f}")
print(f"median-over-positions per request, then median over requests: {np.median([np.median(ad[a][p]) for a in C[1:] for p in range(P)]):.4f}")
rows = []
for a in C[1:]:
    for p in range(P):
        for t in range(d["n_tokens"]):
            rows.append((float(ad[a][p][t]), a, p, t))
rows.sort(reverse=True)
print("\n10 largest per-token differences (adapter, prompt, position, token id, fp32 logprob, bf16 logprob):")
top = []
for diff, a, p, t in rows[:10]:
    tok = d["tokens"][p][C.index(a)][t]
    f, b = float(own(L32, a)[p][t]), float(own(LBF, a)[p][t])
    print(f"   {a} p{p:2d} t{t:3d} token {tok:6d}  fp32 {f:8.3f}  bf16 {b:8.3f}  |diff| {diff:.3f}")
    top.append({"adapter": a, "prompt": p, "pos": t, "token": tok, "fp32": f, "bf16": b})
json.dump(top, open(sys.argv[1].replace("refs_out.json", "top10_positions.json"), "w"))
