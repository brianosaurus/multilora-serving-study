#!/usr/bin/env python3
"""3a analysis (rules fixed in PLAN.md, Amendment 2). Run on the pod: analyze_passes.py. Writes results/raw/analysis_3a.txt content to stdout."""
import json, subprocess, itertools, os, re
PY = "/workspace/envs/vllm/bin/python"; S = "/workspace/study/followup/routing/scripts"
TAG = os.environ.get("PASS_TAG", "passes"); RAW = "results/raw"; REFS = f"{RAW}/refs_prefix.json"; POW = f"{RAW}/refs_prefix_power.json"
R = json.load(open(REFS)); SHARED = [579] * 8 + [554] * 8
def share(calls_or_rows, key_p):
    num = den = 0
    for c in calls_or_rows:
        if c["cached"] is None: return None
        num += min(c["cached"], SHARED[c[key_p]]); den += SHARED[c[key_p]]
    return num / den
out = []
# SGLang
for name in ("sglang_cold", "sglang_warm"):
    f = f"{RAW}/sglang/{TAG}/{name}.json"
    if not os.path.exists(f): out.append(f"{name}: missing"); continue
    d = json.load(open(f))
    r = subprocess.run([PY, f"{S}/analyze_engine.py", REFS, f, POW], capture_output=True, text=True)
    out.append(f"== {name}: errors {d['stats']['errors']}; shared-prefix tokens reported cached: {share(d['calls'], 'p')}")
    out.append(r.stdout.strip() + r.stderr.strip())
# vLLM fallback
nc = [f"{RAW}/vllm/{TAG}/vllm_nc{i}.json" for i in (1, 2, 3)]; wc = f"{RAW}/vllm/{TAG}/vllm_wc2.json"
if all(os.path.exists(f) for f in nc + [wc]):
    key = lambda d: {(r["a"], r["p"]): r["text"] for r in d["complete"]}
    N = [key(json.load(open(f))) for f in nc]; W = json.load(open(wc)); Wk = key(W)
    f0 = sum(sum(x[k] == y[k] for k in x) / len(x) for x, y in itertools.combinations(N, 2)) / 3
    fw = sum(sum(Wk[k] == n[k] for k in Wk) / len(Wk) for n in N) / 3
    mis = []
    for (a, p), t in Wk.items():
        own_all = all(t != n[(a, p)] for n in N)
        if own_all:
            for b in range(4):
                if b != a and all(t == n[(b, p)] for n in N): mis.append((a, p, b))
    sh = share(W["complete"], "p")
    cnt = {}
    for fn in ("metrics_before", "metrics_after"):
        pth = f"{RAW}/vllm/{TAG}/{fn}.txt"
        if os.path.exists(pth):
            for l in open(pth):
                m = re.match(r"(vllm:prefix_cache_\w+_total)\S*\s+([\d.e+]+)", l)
                if m: cnt.setdefault(m[1], {})[fn] = float(m[2])
    out.append(f"== vLLM fallback: f0 (no-cache run vs no-cache run) {f0:.4f}; fw (wc2 vs no-cache runs) {fw:.4f}; margin 1/64 = {1/64:.4f}")
    out.append(f"   cached share of shared prefix in wc2: {sh}; counters: {cnt}")
    out.append(f"   misroutes (completion equals another adapter's in every no-cache run and own in none): {mis}")
    out.append(f"   rule: fw >= f0 - 1/64: {fw >= f0 - 1/64}; cache used (share >= 0.5): {sh is not None and sh >= 0.5}; misroutes 0: {not mis}")
else:
    out.append("vLLM fallback: files missing")
print("\n".join(out))
