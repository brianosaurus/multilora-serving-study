#!/usr/bin/env python3
"""Item D (Amendment 6 and the plan's fixed reading). A first-use difference in a cell = request 1 differs in text from requests 2 to 6 while requests 2 to 6 agree with each other.
Reproducible = appears in at least 2 of 3 starts for the same adapter on the same tree. Counts only; no kernel claim."""
import json, pathlib
D = pathlib.Path(__file__).parent / "results" / "D"
starts = [(1, "old"), (2, "new"), (3, "old"), (4, "new"), (5, "old"), (6, "new")]
R = {(n, t): json.load(open(D / f"start{n}_{t}.json")) for n, t in starts}
names = ["base"] + [f"adapter_{i:04d}" for i in range(8)]
def cell(v):
    r1, rest = v[0], v[1:]; agree = len(set(rest)) == 1
    return dict(first_differs=r1 != rest[0] if agree else None, rest_agree=agree, distinct=len(set(v)))
errs = sum(1 for r in R.values() for v in r.values() for x in v if x.startswith("ERR"))
print("error responses:", errs)
for tree in ("old", "new"):
    print(f"\n== {tree} tree (starts {[n for n,t in starts if t==tree]}) ==")
    print("| cell | " + " | ".join(f"start {n}" for n, t in starts if t == tree) + " | first-use difference in starts | reproducible |\n|---|---|---|---|---|")
    hits_total = 0
    for nm in names:
        row, hits = [], 0
        for n, t in starts:
            if t != tree: continue
            v = R[(n, t)][nm]
            if nm == "base": v = v  # 3 requests: compare request 1 with 2 and 3
            c = cell(v) if len(v) >= 3 else None
            if c["first_differs"]: hits += 1
            row.append(("FIRST-USE DIFF" if c["first_differs"] else ("rest differ" if not c["rest_agree"] else "same")) + f" ({c['distinct']} distinct)")
        print(f"| {nm} | " + " | ".join(row) + f" | {hits} of 3 | {'yes' if hits>=2 else 'no'} |")
        hits_total += hits > 0
    print(f"cells with a first-use difference in at least one start: {hits_total} of 9")
print("\n== second round (3 requests per adapter, after 6 each); any cell where the 3 differ ==")
for n, t in starts:
    bad = [nm for nm in names[1:] if len(set(R[(n, t)][nm + "_round2"])) > 1]
    cross = [nm for nm in names[1:] if R[(n, t)][nm + "_round2"][0] != R[(n, t)][nm][1]]
    print(f"start {n} {t}: round 2 not self-consistent: {bad or 'none'}; round 2 differs from round 1 steady output: {cross or 'none'}")
print("\n== first request of each cell vs the steady output, by tree: how many distinct texts across the 3 starts of a tree per adapter ==")
for tree in ("old", "new"):
    out = []
    for nm in names:
        s = {R[(n, t)][nm][0] for n, t in starts if t == tree}; out.append(f"{nm}:{len(s)}")
    print(tree, " ".join(out))
print("\n== old vs new, steady output (request 2) per adapter, by start pair ==")
for nm in names:
    o = {R[(n, t)][nm][1] for n, t in starts if t == "old"}; w = {R[(n, t)][nm][1] for n, t in starts if t == "new"}
    print(nm, "old distinct", len(o), "new distinct", len(w), "same set" if o == w else "sets differ")
