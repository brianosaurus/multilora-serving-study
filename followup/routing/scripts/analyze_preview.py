#!/usr/bin/env python3
"""POST-HOC PREVIEW (cannot certify anything): the four Gate A1 power checks with the per-request distance computed as
the MEDIAN over positions of |logp_E - logp_ref| instead of the mean. Rule, k = 3 and the four checks are unchanged."""
import json, sys, itertools, statistics as st
import numpy as np
K = 3.0
d = json.load(open(sys.argv[1])); C = d["candidates"]; P = len(d["prompts"]); AGG = sys.argv[2]
agg = {"mean": np.mean, "median": np.median, "trim10": lambda a: float(np.mean(np.sort(a)[len(a)//10: -(len(a)//10)]))}[AGG]
L32 = {v: np.array(d["L32"][v]) for v in C}; LBF = {v: np.array(d["Lbf"][v]) for v in C}
REF = {c: np.array([L32[c][p][i] for p in range(P)]) for i, c in enumerate(C)}
dist = lambda L, v, p, ci: float(agg(np.abs(L[v][p][ci] - REF[C[ci]][p])))
ADP = C[1:]
noise = st.median([dist(LBF, a, p, C.index(a)) for a in ADP for p in range(P)])
def verdict(L, served, name, p):
    right = dist(L, served, p, C.index(name)); wrong = min(dist(L, served, p, ci) for ci, c in enumerate(C) if c != name)
    return right, wrong, right < wrong and wrong - right > K * noise
r1 = [verdict(LBF, a, a, p) for a in ADP for p in range(P)]
r2 = [verdict(LBF, j, i, p) for i in ADP for j in ADP if i != j for p in range(P)]
r3 = [verdict(LBF, "base", a, p) for a in ADP for p in range(P)]
near = sum(x[0] < x[1] for x in r1)
pairs = []
for c1, c2 in itertools.combinations(C, 2):
    d12 = st.median([float(agg(np.abs(L32[c1][p][C.index(c2)] - REF[c2][p]))) for p in range(P)])
    d21 = st.median([float(agg(np.abs(L32[c2][p][C.index(c1)] - REF[c1][p]))) for p in range(P)])
    pairs.append((c1, c2, min(d12, d21)))
ok4 = sum(m > K * noise for _, _, m in pairs)
print(f"aggregate over positions: {AGG}")
print(f"noise {noise:.4f}  threshold k*noise {K*noise:.4f}")
print(f"check 1 (correct stand-in routed correctly, need 64/64): {sum(x[2] for x in r1)}/64   [nearest candidate is right: {near}/64; margin min {min(x[1]-x[0] for x in r1):.3f} median {st.median([x[1]-x[0] for x in r1]):.3f}]")
print(f"check 2 (swapped adapters pass, need 0/192): {sum(x[2] for x in r2)}/192")
print(f"check 3 (base under an adapter name passes, need 0/64): {sum(x[2] for x in r3)}/64")
print(f"check 4 (pairs above k*noise, need 10/10): {ok4}/10   weakest pair {min(pairs, key=lambda x: x[2])[0]} vs {min(pairs, key=lambda x: x[2])[1]} at {min(m for _,_,m in pairs):.3f}")
