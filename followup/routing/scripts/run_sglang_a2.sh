#!/usr/bin/env bash
# A2: SGLang 0.5.23 engine runs at scale 0.02 (plan: followup/routing/PLAN.md, A2 implementation notes)
cd /root/k/routing; L=sglang_a2.log; : > $L; PY=/root/e/sglang/bin/python
say() { echo "$(date -u +%T) $*" >> $L; }
state() { echo "load $(cut -d' ' -f1-3 /proc/loadavg) | gpu $(nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw,utilization.gpu,memory.used --format=csv,noheader | tr -d ' ')"; }
say "host $(hostname); GPU procs: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l); $(state)"
[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { say "REFUSED: GPU not idle"; echo SGLANG_A2_DONE >> $L; exit 1; }
A=/root/k/routing/adapters_s0.02
export PATH=/root/e/sglang/bin:$PATH CUDA_HOME=/usr/local/cuda
CMD=($PY -m sglang.launch_server --model-path /root/k/model --served-model-name qwen --port 30000 --mem-fraction-static 0.6 --context-length 4096
     --enable-lora --max-lora-rank 16 --max-loras-per-batch 16 --lora-target-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
     --disable-radix-cache --lora-paths)
for i in 0 1 2 3; do CMD+=("adapter_000$i=$A/adapter_000$i"); done
"${CMD[@]}" > sglang_server_a2.log 2>&1 &
SP=$!
for i in $(seq 1 120); do [ "$(curl -s -o /dev/null -w %{http_code} -m 3 localhost:30000/health)" = 200 ] && break; kill -0 $SP 2>/dev/null || break; sleep 5; done
say "sglang up? $(curl -s -o /dev/null -w %{http_code} -m 3 localhost:30000/health); $(state)"
$PY engine_probe.py --engine sglang --url http://localhost:30000 --refs refs_s0.02.json --mode serial --limit 12 --out sglang_a2_smoke.json >> $L 2>&1
$PY check_smoke.py refs_s0.02.json sglang_a2_smoke.json >> $L 2>&1
ERR=$($PY -c "import json;d=json.load(open('sglang_a2_smoke.json'));print(d['stats']['errors']+sum(1 for c in d['calls'] if c['logp'] is None or any(x is None for x in c['logp'])))" 2>/dev/null)
say "smoke errors or None values: ${ERR:-NA}"
if [ "${ERR:-1}" != "0" ]; then say "smoke failed: stopping before the full runs"; kill $SP; sleep 8; pkill -9 -f "sglang" 2>/dev/null; echo SGLANG_A2_DONE >> $L; exit 1; fi
for run in "serial sglang_a2_serial" "mixed sglang_a2_mixed" "serial sglang_a2_serial_repeat"; do
  set -- $run; say "start $2: $(state)"
  $PY engine_probe.py --engine sglang --url http://localhost:30000 --refs refs_s0.02.json --mode $1 --out $2.json >> $L 2>&1
  say "end $2: $(state)"
done
kill $SP; sleep 10; pkill -9 -f "sglang" 2>/dev/null; sleep 3
say "server stopped; gpu mem $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
for f in sglang_a2_serial sglang_a2_mixed sglang_a2_serial_repeat; do $PY analyze_engine.py refs_s0.02.json $f.json >> sglang_a2_analysis.txt 2>&1; echo >> sglang_a2_analysis.txt; done
$PY - >> sglang_a2_analysis.txt <<PY
import json, numpy as np
a=json.load(open("sglang_a2_serial.json")); b=json.load(open("sglang_a2_serial_repeat.json"))
da={(c["p"],c["a"],c["c"]):np.array(c["logp"]) for c in a["calls"]}; db={(c["p"],c["a"],c["c"]):np.array(c["logp"]) for c in b["calls"]}
m=[float(np.mean(np.abs(da[k]-db[k]))) for k in da]
print(f"self-consistency (context, not part of the rule): SGLang serial pass vs its repeat over {len(m)} calls: mean |diff| median {np.median(m):.4f}, p95 {np.percentile(m,95):.4f}, max {max(m):.4f}; stand-in noise 0.0201")
PY
say "SGLANG_A2_DONE"; echo SGLANG_A2_DONE >> $L
