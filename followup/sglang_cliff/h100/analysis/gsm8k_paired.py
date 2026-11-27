#!/usr/bin/env python3
"""POST-HOC paired analysis of GSM8K runs (not part of any pre-registered plan).

For each pair of runs, matches problems by id, counts problems correct only in A and only
in B, and computes an exact two-sided McNemar (binomial, p=0.5) p-value on the discordant
pairs. Reads sgleval/*/output-rs0.jsonl ('symbolic_correct' per problem). Standard library only.
Run: python3 gsm8k_paired.py   (writes gsm8k_paired.json, prints tables)
Runs: RTX 4090 Z1_stock, Z1_cg256, Z6_cg256b, AA0_fix (phase2_runpod_64bba) and
H100 tp1_stock, tp1_patched. Withdrawn split-decode variants are excluded.
"""
import glob, json, os, itertools
from math import comb
HERE = os.path.dirname(os.path.abspath(__file__))
SG = os.path.join(HERE, "..", "..", "results", "phase2_runpod_64bba")
H = os.path.join(HERE, "..", "results")
RUNS = {"4090 Z1_stock": f"{SG}/Z1_stock", "4090 Z1_cg256": f"{SG}/Z1_cg256",
        "4090 Z6_cg256b": f"{SG}/Z6_cg256b", "4090 AA0_fix": f"{SG}/AA0_fix",
        "H100 tp1_stock": f"{H}/tp1_stock_gsm8k", "H100 tp1_patched": f"{H}/tp1_patched_gsm8k"}

def load(d):
    f = glob.glob(f"{d}/sgleval/*/output-rs0.jsonl")[0]
    return {r["id"]: bool(r["symbolic_correct"]) for r in map(json.loads, open(f))}

def mcnemar(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)

R = {k: load(v) for k, v in RUNS.items()}
out = {"runs": {}, "pairs": []}
print(f"{'run':18s} n    correct  acc")
for k, r in R.items():
    c = sum(r.values()); out["runs"][k] = {"n": len(r), "correct": c, "accuracy": c / len(r)}
    print(f"{k:18s} {len(r)}  {c}  {100*c/len(r):.2f}%")
print(f"\n{'A':18s} {'B':18s} matched A-only B-only  p (exact McNemar, two-sided)")
for a, b in itertools.combinations(R, 2):
    ids = sorted(set(R[a]) & set(R[b]))
    ao = sum(R[a][i] and not R[b][i] for i in ids); bo = sum(R[b][i] and not R[a][i] for i in ids)
    p = mcnemar(ao, bo)
    out["pairs"].append({"A": a, "B": b, "matched": len(ids), "A_only": ao, "B_only": bo, "p": p})
    print(f"{a:18s} {b:18s} {len(ids):5d} {ao:6d} {bo:6d}  {p:.3g}")
json.dump(out, open(os.path.join(HERE, "gsm8k_paired.json"), "w"), indent=1)
