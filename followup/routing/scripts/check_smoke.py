import json, sys, numpy as np
R = json.load(open(sys.argv[1])); P = json.load(open(sys.argv[2])); C = R["candidates"]
print("calls", len(P["calls"]), "stats", P["stats"])
for c in P["calls"]:
    lp = c["logp"]
    if lp is None:
        print(c["p"], c["a"], c["c"], "None"); continue
    ref = np.array(R["L32"][C[c["c"]]][c["p"]][c["c"]])
    d = np.mean(np.abs(np.array(lp) - ref))
    tag = "RIGHT adapter" if c["c"] == c["a"] + 1 else ""
    print(f"prompt {c['p']:2d} served adapter {c['a']} candidate {c['c']}: n={len(lp)} d={d:.4f} {tag}")
