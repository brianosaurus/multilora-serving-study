#!/usr/bin/env bash
# Amendment 7: SGLang main, no LoRA, two starts (radix cache on, then --disable-radix-cache). Run only when the GPU is idle (after Item D).
D=/root/msf_study/followup/pod_items/results/E_cold_pass; mkdir -p $D; L=$D/run.log; : > $L
say() { echo "$(date -u +%T) $*" >> $L; }
E=/root/envs/sglang_main; M=/dev/shm/models/Qwen2.5-1.5B-Instruct; P=/root/msf_study/differential/results/prompts/prompts_main.jsonl
for mode in radix_on radix_off; do
  for i in $(seq 1 24); do [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] && break; sleep 5; done
  [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { say "REFUSED $mode: GPU not idle"; continue; }
  say "host $(hostname) start $mode; load $(cut -d' ' -f1-3 /proc/loadavg); external CPU: $(python3 /root/msf_study/followup/kernel_rerun/extload.py check 10 2>&1 | tail -1)"
  X=""; [ $mode = radix_off ] && X="--disable-radix-cache"
  setsid nohup env PATH=$E/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda $E/bin/python -m sglang.launch_server --model-path $M --served-model-name qwen --port 30000 --mem-fraction-static 0.85 --context-length 8192 $X > $D/server_$mode.log 2>&1 < /dev/null & SP=$!
  ok=0; for i in $(seq 1 1200); do [ "$(curl -s -o /dev/null -w %{http_code} -m 3 localhost:30000/health)" = 200 ] && { ok=1; break; }; kill -0 $SP 2>/dev/null || break; sleep 3; done
  say "$mode ready=$ok"
  [ $ok = 1 ] && { $E/bin/python /root/msf_study/followup/pod_items/cold_pass.py $D/$mode.json $P >> $D/result.txt 2>&1; say "$mode requests exit $?"; }
  kill $SP 2>/dev/null; sleep 8; pkill -9 -f "[s]glang.launch_server|[s]glang::" 2>/dev/null; sleep 5
done
echo DONE >> $L
