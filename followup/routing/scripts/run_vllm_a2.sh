#!/usr/bin/env bash
# A2: vLLM 0.34.0 engine runs at scale 0.02 (plan: followup/routing/PLAN.md, A2 implementation notes)
cd /root/k/routing; L=vllm_a2.log; : > $L; PY=/root/e/vllm/bin/python
say() { echo "$(date -u +%T) $*" >> $L; }
state() { echo "load $(cut -d' ' -f1-3 /proc/loadavg) | gpu $(nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw,utilization.gpu,memory.used --format=csv,noheader | tr -d ' ')"; }
say "host $(hostname); GPU procs: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l); $(state)"
[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { say "REFUSED: GPU not idle"; echo VLLM_A2_DONE >> $L; exit 1; }
A=/root/k/routing/adapters_s0.02
CMD=(/root/e/vllm/bin/vllm serve /root/k/model --served-model-name qwen --port 8000 --max-model-len 4096 --gpu-memory-utilization 0.85 --enable-lora --max-loras 16 --max-cpu-loras 64 --max-lora-rank 16 --lora-modules)
for i in 0 1 2 3; do CMD+=("adapter_000$i=$A/adapter_000$i"); done
"${CMD[@]}" > vllm_server_a2.log 2>&1 &
SP=$!
for i in $(seq 1 120); do [ "$(curl -s -o /dev/null -w %{http_code} -m 3 localhost:8000/health)" = 200 ] && break; sleep 5; done
say "vllm up; $(state)"
for run in "serial vllm_a2_serial" "mixed vllm_a2_mixed" "serial vllm_a2_serial_repeat"; do
  set -- $run; say "start $2: $(state)"
  $PY engine_probe.py --engine vllm --url http://localhost:8000 --refs refs_s0.02.json --mode $1 --out $2.json >> $L 2>&1
  say "end $2: $(state)"
done
kill $SP; sleep 8; pkill -9 -f "EngineCore" 2>/dev/null; sleep 3
say "server stopped; gpu mem $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
for f in vllm_a2_serial vllm_a2_mixed vllm_a2_serial_repeat; do $PY analyze_engine.py refs_s0.02.json $f.json >> vllm_a2_analysis.txt 2>&1; echo >> vllm_a2_analysis.txt; done
$PY - >> vllm_a2_analysis.txt <<PY
import json, numpy as np
a=json.load(open("vllm_a2_serial.json")); b=json.load(open("vllm_a2_serial_repeat.json"))
da={(c["p"],c["a"],c["c"]):np.array(c["logp"]) for c in a["calls"]}; db={(c["p"],c["a"],c["c"]):np.array(c["logp"]) for c in b["calls"]}
m=[float(np.mean(np.abs(da[k]-db[k]))) for k in da]
print(f"self-consistency (context, not part of the rule): vLLM serial pass vs its repeat over {len(m)} calls: mean |diff| median {np.median(m):.4f}, p95 {np.percentile(m,95):.4f}, max {max(m):.4f}; stand-in noise 0.0201")
PY
say "VLLM_A2_DONE"
echo VLLM_A2_DONE >> $L
