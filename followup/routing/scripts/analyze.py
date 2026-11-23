#!/usr/bin/env python3
"""Decision rule and power checks for the routing follow-up (plan: ../PLAN.md). Input: refs.py output."""
import json, sys, itertools, statistics as st
import numpy as np
K = 3.0
d = json.load(open(sys.argv[1]))
C = d["candidates"]; P = len(d["prompts"])
L32 = {v: np.array(d["L32"][v]) for v in C}      # [p][c][t]
LBF = {v: np.array(d["Lbf"][v]) for v in C}
REF = {c: np.array([L32[c][p][i] for p in range(P)]) for i, c in enumerate(C)}   # fp32 candidate c on its own tokens: [p][t]
def dist(L, v, p, ci):   # served variant v on candidate ci's tokens vs fp32 reference of candidate ci
    return float(np.mean(np.abs(L[v][p][ci] - REF[C[ci]][p])))
ADP = C[1:]
noise_vals = [dist(LBF, a, p, C.index(a)) for a in ADP for p in range(P)]
noise = st.median(noise_vals)
noise_by = {a: st.median([dist(LBF, a, p, C.index(a)) for p in range(P)]) for a in ADP}
print(f"noise (median of {len(noise_vals)} bf16-vs-fp32 distances) = {noise:.4f}; by adapter {({a: round(v,4) for a,v in noise_by.items()})}")
print(f"k = {K}; margin threshold k*noise = {K*noise:.4f}")
def verdict(L, served, name, p):
    """request (p, adapter `name`) answered by variant `served`"""
    right = dist(L, served, p, C.index(name))
    wrong = min(dist(L, served, p, ci) for ci, c in enumerate(C) if c != name)
    return right, wrong, (right < wrong and wrong - right > K * noise)
# check 1: correct implementation passes
r1 = [verdict(LBF, a, a, p) for a in ADP for p in range(P)]
ok1 = sum(x[2] for x in r1)
print(f"\nCHECK 1 (bf16 stand-in, correct routing, must pass all): routed correctly {ok1}/{len(r1)}; margin min {min(x[1]-x[0] for x in r1):.4f} median {st.median([x[1]-x[0] for x in r1]):.4f}; d_right median {st.median([x[0] for x in r1]):.4f} p95 {np.percentile([x[0] for x in r1],95):.4f}")
# check 2: swapped adapters are detected
r2 = [verdict(LBF, j, i, p) for i in ADP for j in ADP if i != j for p in range(P)]
bad2 = sum(x[2] for x in r2)
print(f"CHECK 2 (adapter j served under name i, must fail all): routed 'correctly' {bad2}/{len(r2)} (must be 0)")
# check 3: base under an adapter name is detected
r3 = [verdict(LBF, "base", a, p) for a in ADP for p in range(P)]
bad3 = sum(x[2] for x in r3)
print(f"CHECK 3 (base served under an adapter name, must fail all): routed 'correctly' {bad3}/{len(r3)} (must be 0)")
# check 4: separation of candidate references
print("CHECK 4 (median over prompts of fp32 variant c1 vs reference of c2 on c2's tokens; unordered pair judged by the smaller direction):")
sep_ok = True; rows = []
for c1, c2 in itertools.combinations(C, 2):
    d12 = st.median([float(np.mean(np.abs(L32[c1][p][C.index(c2)] - REF[c2][p]))) for p in range(P)])
    d21 = st.median([float(np.mean(np.abs(L32[c2][p][C.index(c1)] - REF[c1][p]))) for p in range(P)])
    m = min(d12, d21); ok = m > K * noise; sep_ok &= ok
    rows.append((c1, c2, d12, d21, m, ok))
    print(f"   {c1:13s} {c2:13s} {d12:7.4f} {d21:7.4f}  smaller {m:7.4f}  {'ok' if ok else 'BELOW k*noise'}")
print()
checks = {"check1": ok1 == len(r1), "check2": bad2 == 0, "check3": bad3 == 0, "check4": sep_ok}
print("POWER CHECKS:", checks, "=> engine verdicts", "ALLOWED" if all(checks.values()) else "NOT ALLOWED (inconclusive)")
json.dump({"noise": noise, "k": K, "noise_by": noise_by, "checks": checks,
           "check1": {"routed": ok1, "n": len(r1)}, "check2_routed_correctly": bad2, "check3_routed_correctly": bad3,
           "check4": [dict(zip(["c1","c2","d12","d21","smaller","ok"], r)) for r in rows]}, open(sys.argv[1].replace(".json", "_power.json"), "w"), indent=1)
