#!/usr/bin/env python3
"""Build the data table and the four H100 figures for the paper from the raw results.

Reads ../results/*/ (loadgen JSON and server.log files copied from pod f70961d23335),
writes figure_data.json and figures/figA..figD.svg. Standard library only.
Run: python3 make_figures.py
Notes: `tok/s` in the raw files (output_tokens_per_s) counts non-empty streamed text
chunks per second, not tokenizer tokens; the figures label it "chunks/s".
Each saturating cell is ONE run (one server start); the latency cells are the median of
3 starts, with every start drawn. No error bars are drawn where there is one run.
"""
import glob, json, os, re, statistics as S

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

def lat(tp, b, rate):
    v = [json.load(open(f))["summary"] for f in sorted(glob.glob(f"{RES}/tp{tp}_{b}_lat*/zipf32_r{rate}.json"))]
    p95 = [x["ttft_ms"]["p95"] for x in v]
    return {"p95_each": p95, "p95_median": S.median(p95), "n": len(p95)}

def sat(tp, b):
    f = f"{RES}/tp{tp}_{b}_sat/sat_c1024.json"
    if not os.path.exists(f):
        return None
    s = json.load(open(f))["summary"]
    d = {"chunks_per_s": s["output_tokens_per_s"], "requests_per_s": s["requests_per_s"],
         "ttft_p50_ms": s["ttft_ms"]["p50"], "ttft_p95_ms": s["ttft_ms"]["p95"],
         "tpot_p50_ms": s["tpot_ms"]["p50"], "ok": s["ok"], "errors": s["errors"]}
    run = []; graph = []; q = []
    for l in open(f"{RES}/tp{tp}_{b}_sat/server.log"):
        m = re.search(r"Decode batch.*#running-req: (\d+).*cuda graph: (\w+).*#queue-req: (\d+)", l)
        if m:
            run.append(int(m[1])); graph.append(m[2] == "True"); q.append(int(m[3]))
    if run:
      d["log"] = {"decode_lines": len(run), "max_running": max(run), "lines_running_gt_256": sum(x > 256 for x in run),
                "lines_graph_false": graph.count(False), "max_queue": max(q)}
    return d

D = {}
for tp in (1, 2):
    for b in ("stock", "patched", "vllm"):
        D[f"tp{tp}_{b}"] = {"r20": lat(tp, b, 20), "r40": lat(tp, b, 40), "sat": sat(tp, b)}
json.dump(D, open(os.path.join(HERE, "figure_data.json"), "w"), indent=1)

# ---------------------------------------------------------------- svg helpers
MONO = "ui-monospace,monospace"; SANS = "Inter,Helvetica,Arial,sans-serif"
GREY, PUR, INK, MUT, GRID = "#9a9a94", "#7b2cbf", "#111111", "#5a5a55", "#d0d0cc"

def t(x, y, s, size=10.5, anchor="start", fill=INK, font=MONO, weight=None):
    w = f' font-weight="{weight}"' if weight else ""
    s = s.replace("&", "&amp;")
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" fill="{fill}" font-family="{font}"{w}>{s}</text>'

def wrap(s, n):
    out, cur = [], ""
    for w in s.split():
        if len(cur) + len(w) + 1 > n and cur:
            out.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    return out + ([cur] if cur else [])

def svg(w, h, ident, title, desc, body):
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-labelledby="{ident}-t {ident}-d">'
            f'<title id="{ident}-t">{title}</title><desc id="{ident}-d">{desc}</desc>{body}</svg>')

def panel(x0, y0, w, h, ymax, ticks, fmt, groups, ident_unit, ref=None, inside=False):
    """groups: list of (label, [(value, color, each_points_or_None)]) ; zero-based bars."""
    o = []
    sc = lambda v: y0 + h - v / ymax * h
    for tk in ticks:
        o.append(f'<line x1="{x0}" y1="{sc(tk):.1f}" x2="{x0+w}" y2="{sc(tk):.1f}" stroke="{GRID}"/>')
        o.append(t(x0 - 6, sc(tk) + 3.5, fmt(tk), 10, "end", MUT))
    if ref:
        o.append(f'<line x1="{x0}" y1="{sc(ref[0]):.1f}" x2="{x0+w}" y2="{sc(ref[0]):.1f}" stroke="{PUR}" stroke-dasharray="5 4"/>')
        o.append(t(x0 + w, sc(ref[0]) - 4, ref[1], 10, "end", PUR))
    gw = w / len(groups); bw = min(46, gw / 2 - 10)
    for gi, (gl, bars) in enumerate(groups):
        cx = x0 + gw * (gi + .5)
        for bi, (v, col, pts, lab) in enumerate(bars):
            bx = cx + (bi - len(bars) / 2) * (bw + 6) + 3
            o.append(f'<rect x="{bx:.1f}" y="{sc(v):.1f}" width="{bw:.1f}" height="{y0+h-sc(v):.1f}" fill="{col}"/>')
            if pts:
                for p in pts:
                    o.append(f'<circle cx="{bx+bw/2:.1f}" cy="{sc(p):.1f}" r="3" fill="#ffffff" stroke="{INK}" stroke-width="1.2"/>')
            if inside:
                o.append(t(bx + bw / 2, y0 + h - 8, lab, 10.5, "middle", "#ffffff"))
            else:
                o.append(t(bx + bw / 2, sc(max([v] + (pts or []))) - 7, lab, 10.5, "middle"))
        o.append(t(cx, y0 + h + 15, gl, 10.5, "middle", MUT))
    o.append(f'<line x1="{x0}" y1="{y0+h}" x2="{x0+w}" y2="{y0+h}" stroke="{INK}"/>')
    return "".join(o)

def pct(a, b):
    return f"{(b/a-1)*100:+.1f}%".replace("-", "−")

# ---------------------------------------------------------------- Figure A
sA = {k: D[k]["sat"]["chunks_per_s"] for k in ("tp1_stock", "tp1_patched", "tp2_stock", "tp2_patched")}
bodyA = panel(70, 40, 600, 190, 20000, [0, 5000, 10000, 15000, 20000], lambda v: f"{v:,}",
              [("H100 TP=1", [(sA["tp1_stock"], GREY, None, f"{sA['tp1_stock']:,.0f}"), (sA["tp1_patched"], PUR, None, f"{sA['tp1_patched']:,.0f}")]),
               ("H100 TP=2", [(sA["tp2_stock"], GREY, None, f"{sA['tp2_stock']:,.0f}"), (sA["tp2_patched"], PUR, None, f"{sA['tp2_patched']:,.0f}")])], "chunks/s")
bodyA += t(235, 30, pct(sA["tp1_stock"], sA["tp1_patched"]) + " patched", 12, "middle", PUR, SANS, 600)
bodyA += t(535, 30, pct(sA["tp2_stock"], sA["tp2_patched"]) + " patched", 12, "middle", PUR, SANS, 600)
bodyA += f'<rect x="70" y="262" width="10" height="10" fill="{GREY}"/>' + t(86, 271, "stock a9a27e9255 (limit 4,096)", 10.5, "start", MUT)
bodyA += f'<rect x="330" y="262" width="10" height="10" fill="{PUR}"/>' + t(346, 271, "PR #53433 head 9009897b21 (limit 256)", 10.5, "start", MUT)
bodyA += t(70, 290, "output chunks/s, closed loop, concurrency 1,024, 3,000 requests, one run per bar, no error bars", 10, "start", MUT)
figA = svg(700, 300, "figA", "Saturated output throughput on H100, stock against patched SGLang",
           f"Grouped bars, zero-based axis, output chunks per second at closed-loop concurrency 1,024, one run per bar. TP=1: stock {sA['tp1_stock']:,.0f}, patched {sA['tp1_patched']:,.0f} ({pct(sA['tp1_stock'], sA['tp1_patched'])}). TP=2: stock {sA['tp2_stock']:,.0f}, patched {sA['tp2_patched']:,.0f} ({pct(sA['tp2_stock'], sA['tp2_patched'])}).", bodyA)

# ---------------------------------------------------------------- Figure B
r = lambda k: D[k]["r40"]
grp = []
for tp in (1, 2):
    bars = []
    for b, col in (("stock", GREY), ("patched", PUR)):
        x = r(f"tp{tp}_{b}")
        bars.append((x["p95_median"], col, x["p95_each"], f"{int(x['p95_median']+0.5)}"))
    grp.append((f"H100 TP={tp}", bars))
bodyB = panel(70, 36, 600, 190, 800, [0, 200, 400, 600, 800], lambda v: f"{v:,}", grp, "ms", (500, "500 ms pass bar"), inside=True)
bodyB += t(235, 28, pct(r("tp1_stock")["p95_median"], r("tp1_patched")["p95_median"]) + " (medians)", 12, "middle", PUR, SANS, 600)
bodyB += t(535, 28, pct(r("tp2_stock")["p95_median"], r("tp2_patched")["p95_median"]) + " (medians)", 12, "middle", PUR, SANS, 600)
bodyB += f'<rect x="70" y="262" width="10" height="10" fill="{GREY}"/>' + t(86, 271, "stock", 10.5, "start", MUT)
bodyB += f'<rect x="140" y="262" width="10" height="10" fill="{PUR}"/>' + t(156, 271, "patched", 10.5, "start", MUT)
bodyB += '<circle cx="236" cy="267" r="3" fill="#ffffff" stroke="#111111" stroke-width="1.2"/>' + t(246, 271, "one of the 3 server starts; bar = median", 10.5, "start", MUT)
bodyB += t(70, 290, "p95 TTFT, ms, open loop, 40 requests/s offered, 500 measured requests per start", 10, "start", MUT)
figB = svg(700, 300, "figB", "p95 time to first token at 40 requests per second on H100, stock against patched SGLang",
           "Grouped bars with a zero-based axis; dots show each of three server starts. Median p95 TTFT: TP=1 stock %.0f ms, patched %.0f ms; TP=2 stock %.0f ms, patched %.0f ms. A dashed line marks the 500 ms pass bar; every median is below it. The start-to-start spread overlaps between stock and patched." % (
               r("tp1_stock")["p95_median"], r("tp1_patched")["p95_median"], r("tp2_stock")["p95_median"], r("tp2_patched")["p95_median"]), bodyB)

# ---------------------------------------------------------------- Figure C
def cpanel(x0, y0, title, key, ymax, ticks, fmt, f2):
    g = []
    for tp in (1, 2):
        s, p = D[f"tp{tp}_stock"]["sat"], D[f"tp{tp}_patched"]["sat"]
        g.append((f"TP={tp}", [(s[key], GREY, None, f2(s[key])), (p[key], PUR, None, f2(p[key]))]))
    o = t(x0, y0 - 14, title, 11.5, "start", INK, SANS, 600)
    return o + panel(x0 + 44, y0, 250, 120, ymax, ticks, fmt, g, "")
bodyC = (cpanel(10, 40, "Output chunks/s (higher is better)", "chunks_per_s", 20000, [0, 10000, 20000], lambda v: f"{v:,}", lambda v: f"{v:,.0f}")
         + cpanel(360, 40, "Median TTFT, s (lower is better)", "ttft_p50_ms", 8000, [0, 4000, 8000], lambda v: f"{v/1000:.0f}", lambda v: f"{v/1000:.1f}")
         + cpanel(10, 220, "p95 TTFT, s (lower is better)", "ttft_p95_ms", 14000, [0, 7000, 14000], lambda v: f"{v/1000:.0f}", lambda v: f"{v/1000:.1f}")
         + cpanel(360, 220, "Median time per streamed chunk, ms (lower)", "tpot_p50_ms", 50, [0, 25, 50], lambda v: f"{v:.0f}", lambda v: f"{v:.1f}"))
bodyC += f'<rect x="10" y="372" width="10" height="10" fill="{GREY}"/>' + t(26, 381, "stock", 10.5, "start", MUT)
bodyC += f'<rect x="80" y="372" width="10" height="10" fill="{PUR}"/>' + t(96, 381, "patched", 10.5, "start", MUT)
bodyC += t(170, 381, "closed loop, concurrency 1,024, one run per bar, zero-based axes, no error bars", 10, "start", MUT)
sv = lambda tp, b, k: D[f"tp{tp}_{b}"]["sat"][k]
figC = svg(700, 392, "figC", "Saturation trade-off on H100: throughput rises while median time to first token also rises",
           "Four zero-based panels, one run per bar. Chunks/s: TP=1 %.0f to %.0f, TP=2 %.0f to %.0f. Median TTFT: TP=1 %.1f s to %.1f s, TP=2 %.1f s to %.1f s (higher with the patch). p95 TTFT: TP=1 %.1f s to %.1f s, TP=2 %.1f s to %.1f s (lower with the patch). Median time per streamed chunk: TP=1 %.1f to %.1f ms, TP=2 %.1f to %.1f ms." % (
               sv(1,"stock","chunks_per_s"), sv(1,"patched","chunks_per_s"), sv(2,"stock","chunks_per_s"), sv(2,"patched","chunks_per_s"),
               sv(1,"stock","ttft_p50_ms")/1e3, sv(1,"patched","ttft_p50_ms")/1e3, sv(2,"stock","ttft_p50_ms")/1e3, sv(2,"patched","ttft_p50_ms")/1e3,
               sv(1,"stock","ttft_p95_ms")/1e3, sv(1,"patched","ttft_p95_ms")/1e3, sv(2,"stock","ttft_p95_ms")/1e3, sv(2,"patched","ttft_p95_ms")/1e3,
               sv(1,"stock","tpot_p50_ms"), sv(1,"patched","tpot_p50_ms"), sv(2,"stock","tpot_p50_ms"), sv(2,"patched","tpot_p50_ms")), bodyC)

# ---------------------------------------------------------------- Figure D (diagram)
def box(x, y, w, lines, kind):
    """kind: obs (solid, measured or logged), inf (dashed, interpretation), cav (dotted, caveat)"""
    h = 14 * len(lines) + 12
    dash = {"obs": "", "inf": ' stroke-dasharray="5 3"', "cav": ' stroke-dasharray="1.5 3"'}[kind]
    stroke = {"obs": INK, "inf": PUR, "cav": MUT}[kind]
    fill = {"obs": "#ffffff", "inf": "#f6f0fb", "cav": "#fafaf7"}[kind]
    o = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{stroke}"{dash}/>'
    for i, ln in enumerate(lines):
        o += t(x + 10, y + 19 + 14 * i, ln, 10.5, "start", INK if kind == "obs" else (PUR if kind == "inf" else MUT))
    return o, h

def column(x, w, title, steps):
    o = t(x + w / 2, 22, title, 13, "middle", INK, SANS, 600); y = 36; prev = None
    for kind, text in steps:
        b, h = box(x, y, w, wrap(text, int((w - 20) / 6.4)), kind)
        if prev is not None:
            o += f'<line x1="{x+w/2}" y1="{prev}" x2="{x+w/2}" y2="{y-1}" stroke="{MUT}"/><path d="M{x+w/2-4},{y-7} L{x+w/2},{y-1} L{x+w/2+4},{y-7}" fill="none" stroke="{MUT}"/>'
        o += b; prev = y + h; y += h + 16
    return o, y
stock = [("obs", "Running-request limit: 4,096 (server log). Decode CUDA graphs captured to batch size 256 (server log)."),
         ("obs", "Saturating load (1,024 concurrent): logged decode batches reached 799; 15 of 131 sampled decode log lines show a batch above 256, all with cuda graph: False."),
         ("inf", "Interpretation: batches above 256 run eagerly. The eager step cost was measured on the RTX 4090 (about 15 times a replay), not on the H100."),
         ("obs", "Median time per streamed chunk 37.7 ms; median TTFT 2.0 s; 13,141 chunks/s (TP=1)."),
         ("inf", "Interpretation: slower draining can keep batches large. Not isolated on H100.")]
patched = [("obs", "Running-request limit defaults to 256 = largest captured graph batch (server log). Graphs captured to 256 as in stock."),
           ("obs", "Same load: logged decode batches never exceeded 256; all 155 sampled decode log lines show cuda graph: True. Logged queue peaked at 768."),
           ("inf", "Interpretation: requests beyond 256 wait for admission, and active batches can stay in graph coverage."),
           ("obs", "Median time per streamed chunk 11.1 ms; median TTFT 5.4 s; 16,409 chunks/s (TP=1)."),
           ("cav", "Caveat: fitting the captured range does not by itself guarantee replay; other graph eligibility conditions also apply.")]
b1, y1 = column(10, 330, "Stock (H100 TP=1, same pattern at TP=2)", stock)
b2, y2 = column(360, 330, "Patched: limit defaults to graph range", patched)
H = max(y1, y2) + 36
leg = (f'<rect x="10" y="{H-22}" width="14" height="10" rx="2" fill="#ffffff" stroke="{INK}"/>' + t(30, H - 13, "logged or measured", 10, "start", MUT)
       + f'<rect x="170" y="{H-22}" width="14" height="10" rx="2" fill="#f6f0fb" stroke="{PUR}" stroke-dasharray="5 3"/>' + t(190, H - 13, "interpretation", 10, "start", MUT)
       + f'<rect x="290" y="{H-22}" width="14" height="10" rx="2" fill="#fafaf7" stroke="{MUT}" stroke-dasharray="1.5 3"/>' + t(310, H - 13, "caveat", 10, "start", MUT))
figD = svg(700, H, "figD", "Mechanism: running-request limit against decode CUDA graph coverage",
           "Two columns of five boxes each. Stock: running limit 4,096 and graphs to 256; batches above 256 were logged without graphs; interpretation that they run eagerly and slow draining. Patched: running limit 256 equal to the graph range; batches never above 256 and always with graphs in the sampled log lines; extra requests wait. Solid boxes are logged or measured, dashed boxes are interpretation, the dotted box is a caveat.", b1 + b2 + leg)

for n, s in (("figA", figA), ("figB", figB), ("figC", figC), ("figD", figD)):
    open(os.path.join(OUT, n + ".svg"), "w").write(s.replace("<svg ", '<svg xmlns="http://www.w3.org/2000/svg" ', 1))
print("wrote", os.listdir(OUT))
