#!/usr/bin/env python3
"""Phase 2 tables and prediction checks (PLAN.md, Phase 2). Usage: analyze_phase2.py results/phase2"""
import glob, json, os, statistics as st, sys
D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results/phase2")
ADAPTER_BYTES = 36981464


def pct(a, q):
    a = sorted(a); return a[min(len(a) - 1, int(round(q / 100 * (len(a) - 1))))] if a else None


def ranks(x):
    o = sorted(range(len(x)), key=lambda i: x[i]); r = [0.0] * len(x); i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and x[o[j + 1]] == x[o[i]]: j += 1
        for k in range(i, j + 1): r[o[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    if len(x) < 3: return None
    a, b = ranks(x), ranks(y); ma, mb = st.mean(a), st.mean(b)
    num = sum((p - ma) * (q - mb) for p, q in zip(a, b)); den = (sum((p - ma) ** 2 for p in a) * sum((q - mb) ** 2 for q in b)) ** .5
    return num / den if den else None


def load_trace(cell):
    ev = []
    for f in glob.glob(f"{cell}/trace/trace_*.jsonl"):
        for line in open(f):
            try: ev.append(json.loads(line))
            except ValueError: pass
    return ev


res = {}
for cell in sorted(glob.glob(f"{D}/[BDEG]_*")):
    tag = os.path.basename(cell)[2:]; ev = load_trace(cell)
    marks = {}
    for line in open(f"{cell}/marks.jsonl"):
        if line.strip(): m = json.loads(line); marks[int(m["rate"])] = m
    for rate, m in sorted(marks.items()):
        t0, t1 = m["t_start"], m["t_end"]
        arr = [e for e in ev if e["e"] == "arrive" and t0 <= e["t"] <= t1]
        bat = [e for e in ev if e["e"] == "batch" and t0 <= e["t"] <= t1 + 5]
        lod = [e for e in ev if e["e"] == "load" and t0 <= e["t"] <= t1 + 5]
        adm = {}
        for b in sorted(bat, key=lambda e: e["t"]):
            for rid in b.get("rids", []): adm.setdefault(rid, b["t"])
        rows = [(adm[a["rid"]] - a["t"], a) for a in arr if a["rid"] in adm]
        w_res = [w for w, a in rows if a["resident"]]; w_non = [w for w, a in rows if a["resident"] is False]
        busy0 = min((a["t"] for a in arr), default=t0); busy1 = max((b["t"] + b["dur"] for b in bat), default=t1); busy = max(busy1 - busy0, 1e-9)
        dec = [b for b in bat if "DECODE" in b["mode"].upper()]; ext = [b for b in bat if "EXTEND" in b["mode"].upper()]
        s = json.load(open(f"{cell}/zipf32_r{rate}.json"))["summary"]
        nl = sum(len(e["loads"]) for e in lod); ne = sum(len(e["evicts"]) for e in lod)
        res[(tag, rate)] = {
            "arrivals": len(arr), "admitted": len(rows), "busy_s": busy,
            "loads": nl, "evicts": ne, "loads_per_s": nl / busy, "load_MB_per_s": nl * ADAPTER_BYTES / 1e6 / busy,
            "load_time_frac": sum(e["dur"] for e in lod) / busy, "load_calls": len(lod),
            "load_ms_each": 1000 * sum(e["dur"] for e in lod) / nl if nl else None,
            "dec_n_mean": st.mean(b["n"] for b in dec) if dec else None, "dec_adapters_median": st.median(b["adapters"] for b in dec) if dec else None,
            "ext_n_mean": st.mean(b["n"] for b in ext) if ext else None,
            "wait_res_med": st.median(w_res) if w_res else None, "wait_non_med": st.median(w_non) if w_non else None,
            "wait_all_p95": pct([w for w, _ in rows], 95), "wait_all_mean": st.mean(w for w, _ in rows) if rows else None, "n_res": len(w_res), "n_non": len(w_non),
            "frac_wait_gt1": sum(w > 1 for w, _ in rows) / len(rows) if rows else None,
            "spearman_wait_qlen": spearman([w for w, _ in rows], [a["qlen"] for _, a in rows]),
            "gap_ms_mean": 1000 * st.mean(b["gap"] for b in bat if b["gap"] is not None and b["gap"] < 5) if bat else None,
            "tok_per_s": s["output_tokens_per_s"], "ttft_p95_ms": s["ttft_ms"]["p95"], "errors": s["errors"],
            "dec_batch_calls": len(dec)}
f = lambda v, d=2: "n/a" if v is None else (f"{v:.{d}f}" if isinstance(v, float) else str(v))
print("FLAGGED: shared-style screening run, one repeat per cell\n")
keys = ["arrivals", "loads", "evicts", "loads_per_s", "load_MB_per_s", "load_ms_each", "load_time_frac", "dec_n_mean", "dec_adapters_median", "ext_n_mean",
        "n_res", "n_non", "wait_res_med", "wait_non_med", "wait_all_mean", "wait_all_p95", "frac_wait_gt1", "spearman_wait_qlen", "gap_ms_mean", "tok_per_s", "ttft_p95_ms", "errors"]
cols = sorted(res)
print(f"{'':22s}" + "".join(f"{t}@{r}".rjust(16) for t, r in cols))
for k in keys: print(f"{k:22s}" + "".join(f(res[c][k]).rjust(16) for c in cols))
print("\nPrediction checks (A = 16 slots 40 rps, B = 32 slots 40 rps, C = 16 slots 20 rps)")
for v in ("v23", "main"):
    try: A, B, C = res[(f"{v}-16", 40)], res[(f"{v}-32", 40)], res[(f"{v}-16", 20)]
    except KeyError: print(f"  {v}: cells missing"); continue
    ratio = lambda a, b: (a / b) if (a is not None and b) else None
    chk = {
        "H1 loads/s(A)>=3 and >=10x B; B evicts 0; B loads<=40": A["loads_per_s"] >= 3 and (B["loads_per_s"] == 0 or A["loads_per_s"] >= 10 * B["loads_per_s"]) and B["evicts"] == 0 and B["loads"] <= 40,
        "H2 dec adapters(A)>=14 and dec batch(A)<=0.75*B": (A["dec_adapters_median"] or 0) >= 14 and (ratio(A["dec_n_mean"], B["dec_n_mean"]) or 9) <= 0.75,
        "H3 wait non-res(A)>=3s and >=10x res; rho<0.5; B,C waits<0.5s": (A["wait_non_med"] or 0) >= 3 and (A["wait_non_med"] or 0) >= 10 * (A["wait_res_med"] or 1e-9) and abs(A["spearman_wait_qlen"] or 0) < 0.5 and all((x["wait_non_med"] or 0) < 0.5 and (x["wait_res_med"] or 0) < 0.5 for x in (B, C)),
        "H4 load time frac(A)>=0.15 and >=10x B": A["load_time_frac"] >= 0.15 and A["load_time_frac"] >= 10 * max(B["load_time_frac"], 1e-9),
        "H4 not supported if load time frac(A)<0.05": A["load_time_frac"] < 0.05,
        "H5 tok/s(A)<=0.75*B and dec batch(A)<=0.75*B": ratio(A["tok_per_s"], B["tok_per_s"]) is not None and ratio(A["tok_per_s"], B["tok_per_s"]) <= 0.75 and (ratio(A["dec_n_mean"], B["dec_n_mean"]) or 9) <= 0.75,
    }
    print(f"  {v}:")
    for k, ok in chk.items(): print(f"    {'HOLDS ' if ok else 'fails '} {k}")

# vLLM comparison (Part C): queue time from /metrics deltas
import re
cdir = f"{D}/C_vllm-16"
if os.path.isdir(cdir):
    def get(f, name):
        d = {}
        for l in open(f):
            if l.startswith(name):
                k, v = l.rsplit(" ", 1); d[k] = float(v)
        return d
    print("\nvLLM 0.34.0, 16 slots (queue time from vllm:request_queue_time_seconds deltas)")
    for rate in (20, 40):
        b, a = f"{cdir}/metrics_before_r{rate}.txt", f"{cdir}/metrics_after_r{rate}.txt"
        sm = sum(get(a, "vllm:request_queue_time_seconds_sum").values()) - sum(get(b, "vllm:request_queue_time_seconds_sum").values())
        ct = sum(get(a, "vllm:request_queue_time_seconds_count").values()) - sum(get(b, "vllm:request_queue_time_seconds_count").values())
        ba, bb = get(a, "vllm:request_queue_time_seconds_bucket"), get(b, "vllm:request_queue_time_seconds_bucket")
        items = sorted((float(re.search(r'le="([^"]+)"', k).group(1).replace("+Inf", "1e9")), ba[k] - bb.get(k, 0)) for k in ba)
        p95 = next(le for le, c in items if c >= 0.95 * items[-1][1])
        ls = json.load(open(f"{cdir}/zipf32_r{rate}.json"))["summary"]
        print(f"  {rate} rps: mean queue time {sm/ct:.3f} s over {int(ct)} requests, p95 bucket <= {p95} s; loadgen p95 TTFT {ls['ttft_ms']['p95']:.0f} ms, tok/s {ls['output_tokens_per_s']:.0f}")
    for v in ("v23", "main"):
        A = res.get((f"{v}-16", 40))
        if A: print(f"  SGLang {v} 16 slots 40 rps: mean wait {A['wait_all_mean']:.2f} s, p95 {A['wait_all_p95']:.1f} s")
