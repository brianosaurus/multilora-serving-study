#!/usr/bin/env bash
# Item B, Amendment 3: rerun of sglang_s64_n128 only. 60-minute start limit; waits up to 120 s for an idle GPU.
D=/root/msf_study/followup/pod_items/results/B_rerun_sglang_s64_n128; mkdir -p $D; L=$D/run.log; : > $L
M=/dev/shm/models/Qwen2.5-1.5B-Instruct; ENVS=/root/envs; tag=sglang_s64_n128
say() { echo "$(date -u +%T) $*" >> $L; }
for i in $(seq 1 24); do [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] && break; sleep 5; done
[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { say "REFUSED: GPU not idle after 120 s"; exit 1; }
say "host $(hostname); mistralrs build running: $(pgrep -f '[c]argo build' | wc -l) cargo procs; load $(cut -d' ' -f1-3 /proc/loadavg)"
say "external CPU: $(python3 /root/msf_study/followup/kernel_rerun/extload.py check 10 2>&1 | tail -1)"
CMD=($ENVS/sglang/bin/python -m sglang.launch_server --model-path $M --served-model-name qwen --port 30000 --mem-fraction-static 0.85 --context-length 4096 --enable-lora --max-lora-rank 16 --max-loras-per-batch 64 --lora-target-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj --lora-paths)
for n in $(for i in $(seq 0 127); do printf "adapter_%04d " $i; done); do CMD+=("$n=/dev/shm/adapters_128/$n"); done
say "start $tag"
setsid nohup env PATH=$ENVS/sglang/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda "${CMD[@]}" > $D/server_$tag.log 2>&1 < /dev/null & SP=$!
ok=0; for i in $(seq 1 1200); do [ "$(curl -s -o /dev/null -w %{http_code} -m 3 localhost:30000/health)" = 200 ] && { ok=1; break; }; kill -0 $SP 2>/dev/null || break; sleep 3; done
sleep 5; mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)
say "$tag ready=$ok gpu_mem=$mem"; printf "%s\tsglang\t64\t128\t%s\t%s\n" $tag $ok "$mem" > $D/summary.tsv
kill $SP 2>/dev/null; sleep 8; pkill -9 -f "[s]glang.launch_server|[s]glang::" 2>/dev/null; sleep 5
echo DONE >> $L
