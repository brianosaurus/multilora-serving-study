#!/usr/bin/env python3
"""Axis-1 phase 1 table for one model. usage: axis1_phase1.py MODEL RAWDIR REFDIR
Noise floor per engine: R_e = max pairwise D between its 5 repeated baseline runs, S_e = D vs reference on the 16 short prompts,
N_e = max(R_e, S_e, 0.005). Flag rule (PLAN section 5): D > 3*N_e and D-N_e > 0.02 and CI_lo > 1.5*N_e.
Generation-mode cells additionally report: prompts whose whole output equals the reference, and where/how big the first token mismatch is."""
import json, sys, itertools, statistics, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from score import load, per_prompt, cell_stat
M, RAW, REFD = sys.argv[1:4]
ref = load(f"{REFD}/{M}_raw_run1.main.jsonl"); refs = load(f"{REFD}/{M}_raw_run1.short.jsonl"); ref2 = load(f"{REFD}/{M}_raw_run2.main.jsonl"); refh = load(f"{REFD}/{M}_hfdefault_run1.main.jsonl")
print("raw reference repeat (run1 vs run2): identical gens", sum(a["gen_ids"] == b["gen_ids"] for a, b in zip(ref, ref2)), "/", len(ref), "| raw vs hfdefault reference: identical gens", sum(a["gen_ids"] == b["gen_ids"] for a, b in zip(ref, refh)), "/", len(ref))
def self_D(e1, e2):
    ds = []
    for a, b in zip(e1, e2):
        if "gen_ids_engine" in a:
            n = next((i for i in range(min(len(a["gen_ids_engine"]), len(b["gen_ids_engine"]))) if a["gen_ids_engine"][i] != b["gen_ids_engine"][i]), min(len(a["gen_ids_engine"]), len(b["gen_ids_engine"])))
            v = [abs(x - y) for x, y in zip(a["lp_gen_engine"][:n], b["lp_gen_engine"][:n]) if x is not None and y is not None]
        else: v = [abs(x - y) for x, y in zip(a["lp_engine"], b["lp_engine"]) if x is not None and y is not None]
        if v: ds.append(sum(v) / len(v))
    return statistics.fmean(ds) if ds else 0.0
def gen_stats(refrows, rows):
    ident = 0; pos = []; gaps = []; lenonly = 0
    for r, e in zip(refrows, rows):
        g = e.get("gen_ids_engine")
        if g is None: continue
        rg = r["gen_ids"]; n = min(len(g), len(rg)); k = next((i for i in range(n) if g[i] != rg[i]), None)
        if k is None:
            if len(g) == len(rg): ident += 1
            else: lenonly += 1
            continue
        pos.append(k); t = dict((a, b) for a, b in r["top20"][k]); gaps.append(r["lp"][k] - t[g[k]] if g[k] in t else None)
    gp = sorted(x for x in gaps if x is not None)
    return f"identical={ident}/{len(refrows)} len-only={lenonly} token-mismatch={len(pos)} first-mismatch-pos median={statistics.median(pos) if pos else '-'} gap(ref lp - ref lp of engine token) median={statistics.median(gp) if gp else '-':.3} max={gp[-1] if gp else '-'}"
def fmt(s): return "n/a" if not s else f"{s['D']:.4f} [{s['ci_lo']:.4f},{s['ci_hi']:.4f}] n={s['n']} max|d|={s['max_abs']:.2f}"
out = {}
for E in ("vllm", "sglang", "mistralrs"):
    d = f"{RAW}/{M}/{E}"
    if not os.path.isdir(d): continue
    runs = [load(f"{d}/nf_main_run{r}.jsonl") for r in range(1, 6)]
    R = max(self_D(a, b) for a, b in itertools.combinations(runs, 2)); short = cell_stat(per_prompt(refs, load(f"{d}/nf_short.jsonl"))); S = short["D"] if short else 0
    N = max(R, S, 0.005); c0 = cell_stat(per_prompt(ref, runs[0]))
    print(f"\n=== {E}  (method: {'generate-and-compare prefix' if 'gen_ids_engine' in runs[0][0] else 'teacher-forced'})\n  noise floor: R_e(run-to-run)={R:.5f}  S_e(short prompts D vs ref)={S:.5f}  N_e={N:.5f}")
    cells = {"C0 baseline (bs1)": runs[0], "C7 32 concurrent": load(f"{d}/c7_conc32.jsonl"), "C6 cache on (after warm-up)": load(f"{d}/c6_default_pass2.jsonl"), "C6 cache off": load(f"{d}/c6_noprefix_pass2.jsonl")}
    for k, v in cells.items():
        s = cell_stat(per_prompt(ref, v)); flag = ("  FLAG" if (s and s["D"] > 3 * N and s["D"] - N > 0.02 and s["ci_lo"] > 1.5 * N) else "")
        hits = [x.get("cached_tokens") for x in v if x.get("cached_tokens") is not None]
        print(f"  {k:28s} D={fmt(s)}{flag}" + (f"  cached_tokens sum={sum(hits)}" if hits else ""))
    gen_cells = {"C6 gen, cache on": load(f"{d}/c6_default_pass2.jsonl"), "C6 gen, cache off": load(f"{d}/c6_noprefix_pass2.jsonl")}
    for k, v in gen_cells.items(): print(f"  {k:28s} vs raw ref: {gen_stats(ref, v)}")
    c8 = load(f"{d}/c8_defaults.jsonl")
    print(f"  C8 defaults (temp=0 only)   vs raw ref:       {gen_stats(ref, c8)}\n  C8 defaults (temp=0 only)   vs hf-default ref: {gen_stats(refh, c8)}")
    print(f"  C8 neutral (explicit penalty off) = C6 cache-on pass2 vs raw ref above; TF-engines C0 vs raw ref D={c0['D']:.4f}")
    out[E] = {"R": R, "S": S, "N": N}
json.dump(out, open(f"{RAW}/{M}/noise_floor.json", "w"), indent=1)
