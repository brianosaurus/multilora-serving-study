#!/usr/bin/env python3
"""Option 9.2 (PLAN.md): share of decode calls above 48 requests per traced (start, rate) window, batch timeline, tip time. Usage: batch_share.py results/phase2"""
import glob, json, os, statistics as st, sys
D = sys.argv[1] if len(sys.argv) > 1 else "results/phase2"
SLOW20 = {"B_main-16", "E_main-16-P1"}; SLOW40 = {"B_main-16", "E_main-16-U4"}
rows = []
for cell in sorted(glob.glob(f"{D}/[BDEGHK]*_*")):
    tag = os.path.basename(cell)
    if not glob.glob(f"{cell}/trace/*.jsonl") or not os.path.exists(f"{cell}/marks.jsonl"): continue
    ev = [json.loads(l) for f in glob.glob(f"{cell}/trace/*.jsonl") for l in open(f)]
    marks = {int(json.loads(l)["rate"]): json.loads(l) for l in open(f"{cell}/marks.jsonl") if l.strip()}
    for rate, m in sorted(marks.items()):
        t0, t1 = m["t_start"], m["t_end"]
        dec = sorted([e for e in ev if e["e"] == "batch" and "DECODE" in e["mode"].upper() and t0 <= e["t"] <= t1 + 5], key=lambda e: e["t"])
        if not dec: continue
        s = json.load(open(f"{cell}/zipf32_r{rate}.json")); sm = s["summary"]
        reqs = s.get("requests", [])
        slow_req = sorted(r["t_send"] for r in reqs if r.get("ttft_s") is not None and r["ttft_s"] > 1.0)
        share = sum(b["n"] > 48 for b in dec) / len(dec)
        run = best = 0
        for b in dec:
            run = run + 1 if b["n"] > 256 else 0; best = max(best, run)
        # seconds from window start until the batch first stays above 48 for 5 s (1 s buckets, mean n)
        bk = {}
        for b in dec: bk.setdefault(int(b["t"] - t0), []).append(b["n"])
        tl = [st.mean(bk[k]) if k in bk else 0 for k in range(int(t1 - t0) + 1)]
        above = next((k for k in range(len(tl) - 4) if all(x > 48 for x in tl[k:k + 5])), None)
        slow = (rate == 20 and sm["ttft_ms"]["p95"] > 10000) or (rate == 40 and sm["output_tokens_per_s"] < 2500)
        rows.append(dict(tag=tag, rate=rate, slow=slow, share=share, maxn=max(b["n"] for b in dec), run256=best, p95=sm["ttft_ms"]["p95"], tps=sm["output_tokens_per_s"],
                         first_slow_s=(slow_req[0] if slow_req else None), above48_from_s=above, timeline=tl))
print("FLAGGED: existing traced screening starts\n")
print(f"{'start':24s}{'rate':>5s}{'slow':>6s}{'steps>48':>10s}{'max n':>7s}{'run>256':>9s}{'p95 s':>8s}{'tok/s':>7s}{'1st >1s TTFT at':>17s}{'n>48 5s from':>14s}")
for r in rows:
    print(f"{r['tag']:24s}{r['rate']:5d}{'YES' if r['slow'] else '':>6s}{100*r['share']:9.0f}%{r['maxn']:7d}{r['run256']:9d}{r['p95']/1000:8.1f}{r['tps']:7.0f}"
          f"{'' if r['first_slow_s'] is None else format(r['first_slow_s'],'.1f'):>17s}{'' if r['above48_from_s'] is None else str(r['above48_from_s']):>14s}")
for rate in (20, 40):
    sl = [r["share"] for r in rows if r["rate"] == rate and r["slow"]]; nm = [r["share"] for r in rows if r["rate"] == rate and not r["slow"] and r["tag"].split("-")[0] in ("B_main", "E_main", "D_main") and "cg256" not in r["tag"]]
    print(f"\n{rate} rps stock-limit starts: slow n={len(sl)} median share >48 {100*st.median(sl):.0f}%" if sl else f"\n{rate} rps: no slow starts", f"| normal n={len(nm)} median {100*st.median(nm):.0f}% max {100*max(nm):.0f}%" if nm else "")
raised = [r for r in rows if "cg256" in r["tag"]]
print(f"\n(c) raised-limit starts: {len(raised)} windows, longest run of decode calls above 256 = {max(r['run256'] for r in raised)}, max batch {max(r['maxn'] for r in raised)}")
for r in rows:
    if r["slow"]:
        print(f"\ntimeline {r['tag']} @{r['rate']} rps, mean decode batch per second:"); print(" ".join(f"{x:.0f}" for x in r["timeline"][:60]))
