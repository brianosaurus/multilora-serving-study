#!/usr/bin/env bash
# 3a step 1 (per engine, fresh server): 12-call smoke test of cached-token reporting, then the cross-adapter control. No pass is run.
# usage: run_smoke_control.sh vllm|sglang
cd /workspace/study/followup/prefix_routing; . ./serve.sh; E=$1; D=results/raw/$E; mkdir -p $D; L=$D/smoke_control.log; : > $L
PY=/workspace/envs/vllm/bin/python; R=results/raw/refs_prefix.json
say() { echo "$(date -u +%T) $*" >> $L; }
gpu_idle || { say "REFUSED: GPU not idle"; exit 1; }
say "host $(hostname); $(state)"
if [ $E = vllm ]; then start_vllm $D/server_smoke.log --enable-prompt-tokens-details || { say "server did not start"; tail -n 5 $D/server_smoke.log >> $L; exit 1; }; URL=http://localhost:8000
else start_sglang $D/server_smoke.log || { say "server did not start"; tail -n 5 $D/server_smoke.log >> $L; exit 1; }; URL=http://localhost:30000; fi
say "server up; $(state)"
[ $E = vllm ] && curl -s localhost:8000/metrics | grep -E "^vllm:prefix_cache_(hits|queries)" > $D/metrics_before_smoke.txt
$PY prefix_probe.py --engine $E --url $URL --refs $R --mode smoke --out $D/smoke.json >> $L 2>&1
$PY prefix_probe.py --engine $E --url $URL --refs $R --mode control --out $D/control.json >> $L 2>&1
[ $E = vllm ] && curl -s localhost:8000/metrics | grep -E "^vllm:prefix_cache_(hits|queries)" > $D/metrics_after_control.txt
grep -i -E "page.size|block.size" $D/server_smoke.log | head -3 > $D/page_or_block_size.txt
stop_engine; say "stopped; $(state)"
$PY - >> $L <<PY
import json
s=json.load(open("$D/smoke.json")); print("smoke cached:", [c["cached"] for c in s["calls"]], "errors", s["stats"]["errors"])
for r in json.load(open("$D/control.json"))["control"]: print("control", r)
PY
echo SMOKE_CONTROL_DONE >> $L
