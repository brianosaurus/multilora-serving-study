#!/usr/bin/env python3
"""Analysis for run_rerun.sh, applying the establishment rule in PLAN.md (stdlib only).
Usage: analyze_rerun.py OUTDIR BLOCKS [--dry-run]"""
import json, os, sys, statistics as st, re
out, blocks = sys.argv[1], int(sys.argv[2]); dry = "--dry-run" in sys.argv
BINS = ["old", "new", "old2"]; CONCS = [8, 32]; ADS = [0, 1, 4, 8]
def cell(conc, nad, b, blk):
    p = os.path.join(out, "raw", f"c{conc}_a{nad}_{b}_b{blk}.json")
    if not os.path.exists(p): return None
    s = json.load(open(p))["summary"]
    return (s["tpot_ms"]["p50"], s["output_tokens_per_s"], s["errors"]) if s["tpot_ms"]["p50"] is not None else None
flag_lines = [l for l in open(os.path.join(out, "summary.txt")) if "LOAD_ABOVE_LIMIT" in l]
print(("DRY RUN: " if dry else "") + f"{blocks} blocks; runs above the load limit: {len(flag_lines)}")
if flag_lines: print("  flagged runs (first 5):", [" ".join(l.split()[:4]) for l in flag_lines[:5]])
def pct(a, b): return (a - b) / b * 100.0     # change of a relative to b
res = {}
print("\nconc adapters | median ms per token old / new / old2 | new vs old: median change, blocks faster | old2 vs old: median change | tok/s new/old")
for conc in CONCS:
    for nad in ADS:
        rows = {b: [cell(conc, nad, b, k) for k in range(1, blocks + 1)] for b in BINS}
        if any(None in v for v in rows.values()): print(f"{conc:4d} {nad:8d} | missing runs"); continue
        med = {b: st.median(r[0] for r in rows[b]) for b in BINS}
        ch_new = [pct(rows["new"][k][0], rows["old"][k][0]) for k in range(blocks)]
        ch_old2 = [pct(rows["old2"][k][0], rows["old"][k][0]) for k in range(blocks)]
        faster = sum(c < 0 for c in ch_new)
        tps = st.median(r[1] for r in rows["new"]) / st.median(r[1] for r in rows["old"])
        res[(conc, nad)] = dict(ch_new=ch_new, ch_old2=ch_old2, faster=faster, mnew=st.median(ch_new), mold2=st.median(ch_old2))
        print(f"{conc:4d} {nad:8d} | {med['old']:6.2f} {med['new']:6.2f} {med['old2']:6.2f} | {st.median(ch_new):+6.1f}% (per block {', '.join('%+.0f' % c for c in ch_new)}), {faster}/{blocks} | {st.median(ch_old2):+6.1f}% (max abs per block {max(abs(c) for c in ch_old2):.0f}%) | {tps:.2f}x")
print("\nEstablishment rule (PLAN.md): at a cell, (a) the new build is faster in ALL blocks, and (b) the median change is larger in magnitude than the largest median change at the same concurrency in the unchanged-code cells (0 and 1 adapters, new vs old) and in the old-vs-old-copy control (any adapter count).")
for conc in CONCS:
    if any((conc, n) not in res for n in ADS): print(f"concurrency {conc}: incomplete"); continue
    bound = max([abs(res[(conc, n)]["mnew"]) for n in (0, 1)] + [abs(res[(conc, n)]["mold2"]) for n in ADS])
    for nad in (8, 4):
        r = res[(conc, nad)]
        if blocks != 6: verdict = f"NOT EVALUATED (needs 6 blocks, have {blocks})"
        elif flag_lines: verdict = "INVALID: runs exceeded the load limit; not reported as the rerun"
        else:
            a, b = r["faster"] == blocks, abs(r["mnew"]) > bound
            verdict = ("ESTABLISHED" if a and b else "NOT ESTABLISHED") + f" (faster in all blocks: {a}; |median change| {abs(r['mnew']):.1f}% vs bound {bound:.1f}%: {b})"
        print(f"concurrency {conc}, {nad} adapters: {verdict}" + ("   [dry run, discard]" if dry else ""))
