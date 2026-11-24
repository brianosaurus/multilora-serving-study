#!/usr/bin/env bash
# Item C (Amendment 5). SGLang main, no LoRA, default flags, one start.
D=/root/msf_study/followup/pod_items/results/C_repro; mkdir -p $D; L=$D/run.log; : > $L
say() { echo "$(date -u +%T) $*" >> $L; }
for i in $(seq 1 24); do [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] && break; sleep 5; done
[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { say "REFUSED: GPU not idle"; exit 1; }
say "host $(hostname); mistralrs build cargo procs: $(pgrep -fc '[c]argo build'); load $(cut -d' ' -f1-3 /proc/loadavg); external CPU: $(python3 /root/msf_study/followup/kernel_rerun/extload.py check 10 2>&1 | tail -1)"
E=/root/envs/sglang_main; M=/dev/shm/models/Qwen2.5-1.5B-Instruct
setsid nohup env PATH=$E/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda $E/bin/python -m sglang.launch_server --model-path $M --served-model-name qwen --port 30000 --mem-fraction-static 0.85 --context-length 8192 > $D/server.log 2>&1 < /dev/null & SP=$!
say "start"; ok=0; for i in $(seq 1 1200); do [ "$(curl -s -o /dev/null -w %{http_code} -m 3 localhost:30000/health)" = 200 ] && { ok=1; break; }; kill -0 $SP 2>/dev/null || break; sleep 3; done
say "ready=$ok"
if [ $ok = 1 ]; then $E/bin/python /root/msf_study/followup/pod_items/item_c_repro.py /root/msf_study/differential/results/prompts/prompts_main.jsonl $M $D/repro.json > $D/repro.out 2>&1; say "repro exit $?"; fi
grep -iE "sampling.default|default.*sampling|generation_config|preferred" $D/server.log | head -5 >> $L
kill $SP 2>/dev/null; sleep 8; pkill -9 -f "[s]glang.launch_server|[s]glang::" 2>/dev/null; sleep 3; say "stopped"; echo DONE >> $L
