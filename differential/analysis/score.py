#!/usr/bin/env python3
"""Axis-1 scoring. usage: score.py REF_JSONL ENGINE_JSONL -> per-prompt d and cell statistic D with bootstrap CI.
Teacher-forced engines: d = mean |lp_engine - lp_ref| over the G reference gen tokens. mistral.rs (generate fallback): d over the prefix before the first
divergence from the reference tokens, plus the divergence position. Prompts with errors/missing logprobs are counted and excluded."""
import json, sys, random, statistics
def load(p): return [json.loads(l) for l in open(p)]
def per_prompt(ref, eng):
    rows = []
    for r, e in zip(ref, eng):
        row = {"id": r["id"], "G": len(r["gen_ids"]), "err": bool(e.get("error")), "div_pos": None, "d": None, "method": e.get("method")}
        if e.get("error"): rows.append(row); continue
        if "gen_ids_engine" in e:
            ge = e["gen_ids_engine"]; k = next((i for i in range(min(len(ge), len(r["gen_ids"]))) if ge[i] != r["gen_ids"][i]), None)
            if k is None and len(ge) != len(r["gen_ids"]): k = min(len(ge), len(r["gen_ids"]))
            row["div_pos"] = k; n = len(r["gen_ids"]) if k is None else k
            diffs = [abs(e["lp_gen_engine"][i] - r["lp"][i]) for i in range(min(n, len(e["lp_gen_engine"]))) if e["lp_gen_engine"][i] is not None]
        else:
            lp = e.get("lp_engine")
            if lp is None: rows.append(row); continue
            diffs = [abs(a - b) for a, b in zip(lp, r["lp"]) if a is not None]
        if diffs: row["d"] = sum(diffs) / len(diffs); row["max_abs"] = max(diffs); row["n_pos"] = len(diffs)
        rows.append(row)
    return rows
def cell_stat(rows, B=10000, seed=1):
    ds = [x["d"] for x in rows if x["d"] is not None]
    if not ds: return None
    rnd = random.Random(seed); bs = sorted(statistics.fmean(rnd.choices(ds, k=len(ds))) for _ in range(B))
    return {"D": statistics.fmean(ds), "ci_lo": bs[int(.025 * B)], "ci_hi": bs[int(.975 * B)], "n": len(ds), "excluded": len(rows) - len(ds),
            "max_abs": max(x["max_abs"] for x in rows if x["d"] is not None), "div_prompts": sum(1 for x in rows if x["div_pos"] is not None)}
if __name__ == "__main__":
    ref, eng = load(sys.argv[1]), load(sys.argv[2]); rows = per_prompt(ref, eng); print(json.dumps(cell_stat(rows)))
