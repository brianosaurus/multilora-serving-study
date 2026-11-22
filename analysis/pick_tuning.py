#!/usr/bin/env python3
"""Pick each engine's winning trial by the rule in docs/PLAN.md (Tuning protocol).

A rate passes if p95 TTFT is at most 500 ms and there are no errors. A trial's score is the
highest output tokens per second among its passing rates (0 if none pass). The best trial
wins unless it is within 5% of T1, in which case T1 wins.
"""
import glob
import json
import os
import sys

TTFT_LIMIT_MS = 500.0
TIE_BAND = 0.05


def main(path: str) -> None:
    runs = {}
    for f in sorted(glob.glob(os.path.join(path, "*_T?_r*.json"))):
        eng, trial, rate = os.path.basename(f)[:-5].split("_")
        d = json.load(open(f))
        s = d["summary"]
        runs.setdefault(eng, {}).setdefault(trial, {})[int(rate[1:])] = s
    winners = {}
    for eng, trials in runs.items():
        print(f"== {eng}")
        scores = {}
        for t, rates in sorted(trials.items()):
            best = 0.0
            for r, s in sorted(rates.items()):
                ok = s["errors"] == 0 and s["ttft_ms"]["p95"] is not None and s["ttft_ms"]["p95"] <= TTFT_LIMIT_MS
                tps = s["output_tokens_per_s"] or 0.0
                print(f"   {t} rate {r:>3}: p95 TTFT {s['ttft_ms']['p95']:.0f} ms, errors {s['errors']}, "
                      f"{tps:.0f} tok/s  {'PASS' if ok else 'fail'}")
                if ok:
                    best = max(best, tps)
            scores[t] = best
        top = max(scores, key=scores.get)
        win = "T1" if "T1" in scores and scores[top] <= scores["T1"] * (1 + TIE_BAND) else top
        winners[eng] = {"scores_tok_per_s": scores, "winner": win}
        print(f"   scores {scores} -> winner {win}")
    json.dump(winners, open(os.path.join(path, "winners.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
