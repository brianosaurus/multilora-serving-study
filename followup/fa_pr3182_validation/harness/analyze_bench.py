import json, glob, statistics as st, collections, sys
D = collections.defaultdict(dict)  # (mode,tree,block) -> {cfgkey: row}
for f in glob.glob("/root/fa/raw/bench/*_b*_*.json"):
    mode, b, tree = f.split("/")[-1][:-5].split("_"); b = int(b[1:])
    d = json.load(open(f)); D[(mode, tree, b)] = {(r["layout"], r["dtype"], r["V"], r["n"]): r for r in d["rows"]}
def ratios(mode, a, b, metric, key):
    out = []
    for blk in range(10 if mode == "contig" else 6):
        if (mode, a, blk) in D and (mode, b, blk) in D:
            out.append(D[(mode, a, blk)][key][metric]["median"] / D[(mode, b, blk)][key][metric]["median"])
    return out
def line(mode, a, b, metric, key):
    r = ratios(mode, a, b, metric, key); 
    return st.median(r), min(r), max(r), len(r)
keys = sorted({k for (m, t, b), v in D.items() if m == "contig" for k in v}, key=lambda k: (k[1], k[2], k[3]))
print("CONTIGUOUS LABELS: median over blocks of per-block (variant median / stock median); [min..max] across blocks; stock2 is an unchanged-code control")
print("cfg  metric  stock_ms | stock2/stock | patched/stock | alt/stock")
allp = collections.defaultdict(list); allc = collections.defaultdict(list)
for k in keys:
    for metric in ("fwd", "bwd", "fwdbwd"):
        sm = st.median(D[("contig", "stock", b)][k][metric]["median"] for b in range(10))
        cells = []
        for v in ("stock2", "patched", "alt"):
            m, lo, hi, n = line("contig", v, "stock", metric, k); cells.append(f"{m:6.3f} [{lo:.3f}..{hi:.3f}]")
            (allc if v == "stock2" else allp)[(v, metric)].append(m)
        print(f"{k[1]:9s} V{k[2]:<6d} n{k[3]:<6d} {metric:7s} {sm:9.4f} | {cells[0]} | {cells[1]} | {cells[2]}")
print("\nSUMMARY over the 16 configs (median of per-config ratios; min..max of per-config median ratios)")
for (v, metric), xs in sorted({**allc, **allp}.items()):
    print(f"  {v:8s}/stock {metric:7s} median {st.median(xs):.3f}  range {min(xs):.3f}..{max(xs):.3f}")
print("\nSTRIDED LABELS (stock is incorrect here and is not benchmarked): alt(.contiguous()) / patched, per-block median ratio; extra peak memory MB")
skeys = sorted({k for (m, t, b), v in D.items() if m == "strided" for k in v}, key=lambda k: (k[0], k[2], k[3]))
for k in skeys:
    for metric in ("fwd", "bwd", "fwdbwd"):
        pm = st.median(D[("strided", "patched", b)][k][metric]["median"] for b in range(6))
        m, lo, hi, n = line("strided", "alt", "patched", metric, k)
        mem_p = D[("strided", "patched", 0)][k]["extra_peak_bytes_fwdbwd"] / 2**20; mem_a = D[("strided", "alt", 0)][k]["extra_peak_bytes_fwdbwd"] / 2**20
        print(f"{k[0]:4s} {k[1]:8s} V{k[2]:<6d} n{k[3]:<6d} {metric:7s} patched {pm:8.4f} ms  alt/patched {m:6.3f} [{lo:.3f}..{hi:.3f}]  extra peak MB patched {mem_p:8.1f} alt {mem_a:8.1f}")
