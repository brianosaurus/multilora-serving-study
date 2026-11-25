#!/usr/bin/env python3
"""Descriptive summary of the incomplete B2 rerun, blocks 1-4 only (stdlib only).
Usage: analyze_b2_partial.py [OUTDIR] [BLOCKS]. Does not apply the PLAN.md rule."""
import json, os, sys, statistics as st
out = sys.argv[1] if len(sys.argv) > 1 else "results/B2_runpod_INCOMPLETE"
blocks = int(sys.argv[2]) if len(sys.argv) > 2 else 4
def p50(c, a, b, k):
    return json.load(open(os.path.join(out, "raw", f"c{c}_a{a}_{b}_b{k}.json")))["summary"]["tpot_ms"]["p50"]
def ch(c, a, x):  # per-block change of build x vs old
    return [(p50(c, a, x, k) - p50(c, a, "old", k)) / p50(c, a, "old", k) * 100 for k in range(1, blocks + 1)]
print(f"blocks 1-{blocks}; per-block % change in median time per streamed chunk vs old")
for c in (8, 32):
    for a in (0, 1, 4, 8):
        for x in ("new", "old2"):
            v = ch(c, a, x)
            print(f"c{c} a{a} {x:4s} median {st.median(v):+6.1f}%  range {min(v):+.1f} to {max(v):+.1f}  faster {sum(i<0 for i in v)}/{blocks}")
        print(f"c{c} a{a} old median ms {st.median(p50(c,a,'old',k) for k in range(1,blocks+1)):.2f} new {st.median(p50(c,a,'new',k) for k in range(1,blocks+1)):.2f}")
