#!/usr/bin/env bash
# 3a step 2 (per engine). sglang: server restarted, cold mixed probe pass, warm mixed probe pass (same server).
# vllm (fallback, Amendment 2): server with --no-enable-prefix-caching: 3 runs of 'complete'; then a fresh cache-on server: 2 runs of 'complete'.
# usage: [PASS_TAG=name A_DIR=adapters MODEL=model] run_passes.sh vllm|sglang   (PASS_TAG keeps a repeat from overwriting earlier data)
cd /workspace/study/followup/prefix_routing; . ./serve.sh; E=$1; D=results/raw/$E/${PASS_TAG:-passes}; mkdir -p $D; L=$D/passes.log; : > $L
PY=/workspace/envs/vllm/bin/python; R=results/raw/refs_prefix.json
say() { echo "$(date -u +%T) $*" >> $L; }
gpu_idle || { say "REFUSED: GPU not idle"; exit 1; }
say "host $(hostname); $(state)"
run() { $PY prefix_probe.py --engine $E --url $URL --refs $R --out $D/$1.json "${@:2}" >> $L 2>&1; say "end $1: $(state)"; }
if [ $E = sglang ]; then
  start_sglang $D/server.log || { say "server did not start"; exit 1; }; URL=http://localhost:30000; say "server up; $(state)"
  say "start cold"; run sglang_cold --mode mixed
  say "start warm"; run sglang_warm --mode mixed
  stop_engine
else
  start_vllm $D/server_nocache.log --no-enable-prefix-caching || { say "nocache server did not start"; exit 1; }; URL=http://localhost:8000; say "nocache server up; $(state)"
  for i in 1 2 3; do say "start nc$i"; run vllm_nc$i --mode complete; done
  stop_engine
  start_vllm $D/server_cache.log --enable-prompt-tokens-details || { say "cache server did not start"; exit 1; }; say "cache server up; $(state)"
  curl -s localhost:8000/metrics | grep -E "^vllm:prefix_cache_(hits|queries)" > $D/metrics_before.txt
  say "start wc1"; run vllm_wc1 --mode complete
  say "start wc2"; run vllm_wc2 --mode complete
  curl -s localhost:8000/metrics | grep -E "^vllm:prefix_cache_(hits|queries)" > $D/metrics_after.txt
  stop_engine
fi
say "stopped; $(state)"; echo PASSES_DONE >> $L
