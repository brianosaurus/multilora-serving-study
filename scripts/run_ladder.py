#!/usr/bin/env python3
"""Rate ladder for one workload on a running engine (see the Day plan in docs/PLAN.md).

For each rate step it runs REPEATS loadgen runs. A step is a violation if the median p95
TTFT of its runs exceeds the limit. The ladder stops after two consecutive violations. A
step is also flagged load-generator-limited if any run's send lag exceeds MAX_LAG_S, because
the offered rate was then not delivered.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys

TTFT_LIMIT_MS = 500.0
MAX_LAG_S = 0.05
HERE = os.path.dirname(os.path.abspath(__file__))

WORKLOADS = {  # name -> loadgen arguments
    "b0": ["--adapters", "0"],
    "b1": ["--adapters", "0"],
    "one": ["--adapters", "1", "--adapter-offset", "0"],
    "zipf32": ["--adapters", "32", "--adapter-offset", "0", "--popularity", "zipf"],
    "pfx_base": ["--adapters", "0", "--shared-prefix-tokens", "224"],
    "pfx32": ["--adapters", "32", "--adapter-offset", "0", "--popularity", "uniform",
              "--shared-prefix-tokens", "224"],
    "uni32": ["--adapters", "32", "--adapter-offset", "0", "--popularity", "uniform"],
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--url", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--workload", required=True, choices=sorted(WORKLOADS))
    ap.add_argument("--rates", default="5,10,20,40,80")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--requests", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    steps = []
    consecutive = 0
    for rate in [float(r) for r in args.rates.split(",")]:
        runs = []
        for rep in range(args.repeats):
            out = os.path.join(args.outdir, f"{args.engine}_{args.workload}_r{rate:g}_rep{rep}.json")
            cmd = [sys.executable, os.path.join(HERE, "loadgen.py"), "--engine", args.engine, "--url", args.url,
                   "--tokenizer", args.tokenizer, "--rate", str(rate), "--requests", str(args.requests),
                   "--warmup", str(args.warmup), "--seed", str(1 + rep), "--out", out] + WORKLOADS[args.workload]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
            runs.append(json.load(open(out))["summary"])
        p95 = statistics.median(r["ttft_ms"]["p95"] for r in runs if r["ttft_ms"]["p95"] is not None)
        errors = sum(r["errors"] for r in runs)
        lag = max(r["max_send_lag_s"] or 0 for r in runs)
        violated = p95 > TTFT_LIMIT_MS or errors > 0
        step = {"rate": rate, "median_p95_ttft_ms": p95, "errors": errors, "max_send_lag_s": lag,
                "load_generator_limited": lag > MAX_LAG_S, "violation": violated,
                "tok_per_s": [r["output_tokens_per_s"] for r in runs]}
        steps.append(step)
        print(json.dumps(step), flush=True)
        consecutive = consecutive + 1 if violated else 0
        if consecutive >= 2:
            break
    json.dump({"engine": args.engine, "workload": args.workload, "steps": steps},
              open(os.path.join(args.outdir, f"{args.engine}_{args.workload}_ladder.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
