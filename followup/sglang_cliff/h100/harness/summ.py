import json, glob, os, sys
tp = sys.argv[1]
os.chdir("/workspace/h100_run/results")
for tag in sorted(glob.glob(f"tp{tp}_*_lat*")) + sorted(glob.glob(f"tp{tp}_*_sat")):
    for f in sorted(glob.glob(tag + "/*.json")):
        b = os.path.basename(f)
        if b.startswith("warm"):
            continue
        d = json.load(open(f))["summary"]
        print(f"{tag:22s} {b:16s} ok={d['ok']:5d} err={d['errors']} p50={d['ttft_ms']['p50']:8.1f} "
              f"p95={d['ttft_ms']['p95']:8.1f} tpot50={d['tpot_ms']['p50']:5.2f} tok/s={d['output_tokens_per_s']:8.1f} rps={d['requests_per_s']:6.2f}")
for g in sorted(glob.glob(f"tp{tp}_*_gsm8k/sgleval/*/metrics.json")):
    p = json.load(open(g))["aggregate"]["score"]
    print(g.split("/")[0], f"acc={p:.4f} se={(p * (1 - p) / 1319) ** .5:.4f}")
