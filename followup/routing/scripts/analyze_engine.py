#!/usr/bin/env python3
"""A2 analysis: apply the plan's decision rule to engine probe output. Usage: analyze_engine.py REFS PROBE_JSON [POWER_JSON]
noise and k come from the stand-in's power check at the chosen scale (PLAN.md): routed correctly iff d_right < d_wrong and
d_wrong - d_right > k * noise; a request counts only if all 5 candidate calls returned 128 values."""
import json, sys, statistics as st
import numpy as np
K = 3.0
R = json.load(open(sys.argv[1])); P = json.load(open(sys.argv[2])); C = R["candidates"]; ADP = C[1:]
PW = json.load(open(sys.argv[3] if len(sys.argv) > 3 else sys.argv[1].replace(".json", "_power.json"))); noise = PW["noise"]
REF = {c: np.array([R["L32"][c][p][i] for p in range(len(R["prompts"]))]) for i, c in enumerate(C)}   # fp32 candidate c on its own tokens
by = {(c["p"], c["a"], c["c"]): c["logp"] for c in P["calls"]}
rows, bad_req, pending = [], 0, 0
for p in range(len(R["prompts"])):
    for ai, name in enumerate(ADP):
        ds = {}
        for ci in range(5):
            lp = by.get((p, ai, ci))
            if lp is None or len(lp) != R["n_tokens"] or any(x is None for x in lp): continue
            ds[ci] = float(np.mean(np.abs(np.array(lp) - REF[C[ci]][p])))
        if len(ds) < 5: bad_req += 1; continue
        right = ds[ai + 1]; wrong_ci = min((ci for ci in ds if ci != ai + 1), key=lambda ci: ds[ci])
        rows.append((p, name, right, ds[wrong_ci], C[wrong_ci], right < ds[wrong_ci] and ds[wrong_ci] - right > K * noise))
ok = sum(r[5] for r in rows)
print(f"{P['engine']} / {P['mode']}: noise {noise:.4f}, threshold {K*noise:.4f}; requests evaluated {len(rows)}/64 ({bad_req} incomplete)")
print(f"  routed correctly: {ok}/{len(rows)}  ({'PASS' if ok == len(rows) == 64 else 'NOT a pass'} on the 100% rule)")
if rows:
    m = [r[3] - r[2] for r in rows]
    print(f"  margin d_wrong - d_right: min {min(m):.4f} median {st.median(m):.4f}; d_right median {st.median([r[2] for r in rows]):.4f} (stand-in noise {noise:.4f}), max {max(r[2] for r in rows):.4f}")
mis = [r for r in rows if not r[5]]
print(f"  not routed correctly: {len(mis)}" + ("" if not mis else " (first 10, or all if fewer):"))
for p, name, right, wrong, wc, _ in sorted(mis, key=lambda r: r[3] - r[2])[:10]:
    kind = "NEAREST IS WRONG" if wrong < right else "margin too small"
    print(f"    prompt {p:2d} adapter {name}: d_right {right:.4f} d_wrong {wrong:.4f} nearest other candidate {wc}  [{kind}]")
print("  probe stats:", P["stats"])
