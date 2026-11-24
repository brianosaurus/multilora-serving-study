#!/usr/bin/env bash
# 3a step 1 (per engine, KIND, fresh server each): the cross-adapter control with only that kind of call. No smoke, no pass.
# usage: run_smoke_control.sh vllm|sglang probe|plain   (amended 2026-11-24 after the first run was confounded, see PLAN.md)
cd /workspace/study/followup/prefix_routing; . ./serve.sh; E=$1; K=$2; D=results/raw/$E/$K; mkdir -p $D; L=$D/control.log; : > $L
PY=/workspace/envs/vllm/bin/python; R=results/raw/refs_prefix.json
say() { echo "$(date -u +%T) $*" >> $L; }
gpu_idle || { say "REFUSED: GPU not idle"; exit 1; }
say "host $(hostname); $(state)"
if [ $E = vllm ]; then start_vllm $D/server.log --enable-prompt-tokens-details || { say "server did not start"; tail -n 5 $D/server.log >> $L; exit 1; }; URL=http://localhost:8000
else start_sglang $D/server.log || { say "server did not start"; tail -n 5 $D/server.log >> $L; exit 1; }; URL=http://localhost:30000; fi
say "server up; $(state)"
[ $E = vllm ] && curl -s localhost:8000/metrics | grep -E "^vllm:prefix_cache_(hits|queries)" > $D/metrics_before.txt
$PY prefix_probe.py --engine $E --url $URL --refs $R --mode control --kinds $K --out $D/control.json >> $L 2>&1
[ $E = vllm ] && curl -s localhost:8000/metrics | grep -E "^vllm:prefix_cache_(hits|queries)" > $D/metrics_after_control.txt
grep -i -E "page.size|block.size" $D/server.log | head -3 > $D/page_or_block_size.txt
stop_engine; say "stopped; $(state)"
$PY - >> $L <<PY
import json
for r in json.load(open("$D/control.json"))["control"]: print("control", r)
PY
echo CONTROL_DONE >> $L
