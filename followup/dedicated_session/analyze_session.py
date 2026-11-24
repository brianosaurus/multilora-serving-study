#!/usr/bin/env python3
"""analyze_session.py OUT_DIR : tables and the verdicts defined in PLAN.md (written before any run). Reads OUT_DIR/cells.tsv.
A verdict is "not evaluated" when any run involved is flagged (LOAD_ABOVE_LIMIT or LENGTH), its session has drift above 5%, or a run is missing."""
import sys, statistics as st, collections
out = sys.argv[1]
rows, drift, failed = [], {}, set()
for l in open(f"{out}/cells.tsv"):
    f = l.rstrip("\n").split("\t")
    if len(f) < 3: continue
    if f[2] == "DRIFT_CHECK": drift[f[1]] = f[3]; continue
    if f[2] == "START_FAILED": failed.add(f[1]); continue
    if f[1] == "3a_repeat" or len(f) < 12: continue
    rows.append(dict(block=f[0], tag=f[1], wl=f[2], rate=float(f[3]), rep=int(f[4]), flag=f[5], ttft=float(f[6]), e2e=float(f[7]), err=int(float(f[8])), tps=float(f[9]), bad=int(float(f[11]))))
ALLOW = [False]
blocking = lambda f: any(t not in ("OK", "LOAD_ABOVE_LIMIT") for t in f.split("+")) if ALLOW[0] else f != "OK"
cells = collections.defaultdict(list)
for r in rows: cells[(r["tag"], r["wl"], r["rate"])].append(r)
def stat(tag, wl, rate):
    c = cells.get((tag, wl, rate), [])
    if not c: return None
    t = [x["ttft"] for x in c]
    return dict(n=len(c), med=st.median(t), lo=min(t), hi=max(t), tps=st.median(x["tps"] for x in c), flagged=sum(blocking(x["flag"]) for x in c), err=sum(x["err"] for x in c))
def clean(tag, wl, rate, n=3):
    s = stat(tag, wl, rate)
    return s is not None and s["n"] >= n and s["flagged"] == 0 and not drift.get(tag, "ok").endswith("DRIFT") and tag not in failed
P = []
P.append("# Session analysis (definitions in PLAN.md)\n")
P.append("| session | workload | rate | n | p95 TTFT ms median (min-max) | tok/s | flagged | errors |\n|---|---|---|---|---|---|---|---|")
for k in sorted(cells):
    s = stat(*k); P.append(f"| {k[0]} | {k[1]} | {k[2]:g} | {s['n']} | {s['med']:.0f} ({s['lo']:.0f}-{s['hi']:.0f}) | {s['tps']:.0f} | {s['flagged']} | {s['err']} |")
P.append("\nDrift checks: " + "; ".join(f"{t}: {d}" for t, d in sorted(drift.items())) + ("\nServers that failed to start: " + ", ".join(sorted(failed)) if failed else ""))
def verdicts(allow):
    ALLOW[0] = allow
    P.append('\n# ' + ('SHARED-HOST READING (Amendment 8): flagged-for-load runs are used and labeled; LENGTH, drift, failed starts and missing runs still block' if allow else 'PRE-REGISTERED READING: any flagged run blocks a verdict') + '\n')
    def pf(tag, wl, rate, bar=500):
        s = stat(tag, wl, rate); return None if s is None else s["med"] <= bar
    P.append("\n## 4b: does the cliff reproduce at B scale 0.02, and does the slot fix\n(cliff reproduces for an engine and popularity: 16 slots, p95 TTFT median at or under 500 ms at the rate that passed in the main grid and over 500 ms at the next one: Zipf 20 then 40, uniform 10 then 20; slot fix reproduces: Zipf, 32 slots, 40 rps at or under 500 ms)")
    for eng in ("vllm", "sglang"):
        for wl, lo, hi in (("zipf32", 20, 40), ("uni32", 10, 20)):
            t = f"4b_{eng}_s16"; ok = clean(t, wl, lo) and clean(t, wl, hi)
            v = "not evaluated (flagged, drift, or missing runs)" if not ok else ("REPRODUCES" if pf(t, wl, lo) and not pf(t, wl, hi) else "DOES NOT REPRODUCE")
            P.append(f"- cliff, {eng}, {wl}, 16 slots: {v}")
        t = f"4b_{eng}_s32"; ok = clean(t, "zipf32", 40)
        P.append(f"- slot fix, {eng}, zipf32, 32 slots, 40 rps: " + ("not evaluated (flagged, drift, or missing runs)" if not ok else ("REPRODUCES" if pf(t, "zipf32", 40) else "DOES NOT REPRODUCE")))
        s = stat(f"4b_{eng}_s32", "uni32", 20)
        if s: P.append(f"- (no prior) {eng}, uni32, 32 slots, 20 rps: median p95 TTFT {s['med']:.0f} ms, flagged {s['flagged']}")
    P.append("\n## 3b: SGLang main against v0.5.23 (improved: median at least 20% lower and ranges not overlapping; worse: at least 20% higher and not overlapping; else unchanged; baseline must show the cliff: v0.5.23 default 16 slots Zipf 40 rps over 500 ms)")
    base = "3b_sglang_def_s16"
    if not clean(base, "zipf32", 40) or stat(base, "zipf32", 40)["med"] <= 500:
        P.append("- baseline did not reproduce the cliff (or is flagged or missing): the comparison is not interpreted")
    else:
        for variant in ("def", "drain1"):
            b, m = f"3b_sglang_{variant}_s16", f"3b_sglang_main_{variant}_s16"
            for wl, rates in (("zipf32", (20, 40, 80)), ("uni32", (20, 40))):
                for r in rates:
                    if not (clean(b, wl, r) and clean(m, wl, r)): P.append(f"- {variant} {wl} {r:g} rps: not evaluated"); continue
                    x, y = stat(b, wl, r), stat(m, wl, r)
                    v = "improved" if y["med"] <= 0.8 * x["med"] and y["hi"] < x["lo"] else ("worse" if y["med"] >= 1.2 * x["med"] and y["lo"] > x["hi"] else "unchanged")
                    P.append(f"- {variant} {wl} {r:g} rps: v0.5.23 {x['med']:.0f} ms, main {y['med']:.0f} ms: {v}" + (" (main passes 500 ms)" if y["med"] <= 500 < x["med"] else ""))
        b, m = "3b_sglang_def_s32", "3b_sglang_main_def_s32"
        if clean(b, "zipf32", 40) and clean(m, "zipf32", 40):
            x, y = stat(b, "zipf32", 40), stat(m, "zipf32", 40)
            P.append(f"- 32 slots zipf32 40 rps: v0.5.23 {x['med']:.0f} ms, main {y['med']:.0f} ms")
    P.append("\n## 3c: 128 adapters on 64 slots, Zipf (32 slots earlier: vLLM 20 rps 114 ms, 40 rps 3011 ms; SGLang 83 ms and 11662 ms). 'The cliff moved' for an engine: 64-slot p95 TTFT at 40 rps at or under 500 ms")
    for eng in ("vllm", "sglang"):
        t = f"3c_{eng}_s64_n128"
        for r in (20, 40):
            s = stat(t, "zipf128", r)
            P.append(f"- {eng} {r} rps: " + ("missing" if s is None else f"{s['med']:.0f} ms ({s['lo']:.0f}-{s['hi']:.0f}), flagged {s['flagged']}"))
        if clean(t, "zipf128", 40): P.append(f"- {eng}: cliff moved: {pf(t, 'zipf128', 40)}")
verdicts(False)
verdicts(True)
P.append("\n## Flagged runs (reported, not used in verdicts)")
for r in rows:
    if r["flag"] != "OK": P.append(f"- {r['tag']} {r['wl']} {r['rate']:g} rep{r['rep']}: {r['flag']}")
print("\n".join(P))
