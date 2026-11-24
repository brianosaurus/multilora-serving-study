#!/usr/bin/env bash
# Item A check: SGLang main starts with the study's LoRA flags on the 32 adapters; startup log kept; no request is sent.
D=/workspace/study/followup/pod_items/results/A; mkdir -p $D; L=$D/check.log; : > $L
say() { echo "$(date -u +%T) $*" >> $L; }
[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { say "REFUSED: GPU not idle"; exit 1; }
say "host $(hostname); load $(cut -d' ' -f1-3 /proc/loadavg)"
A=/dev/shm/adapters; M=/dev/shm/models/Qwen2.5-1.5B-Instruct
CMD=(/workspace/envs/sglang_main/bin/python -m sglang.launch_server --model-path $M --served-model-name qwen --port 30000 --mem-fraction-static 0.85 --context-length 4096 --enable-lora --max-lora-rank 16 --max-loras-per-batch 16 --lora-target-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj --lora-paths)
for i in $(seq 0 31); do n=$(printf "adapter_%04d" $i); CMD+=("$n=$A/$n"); done
PATH=/workspace/envs/sglang_main/bin:/usr/local/cuda/bin:/root/.cargo/bin:$PATH CUDA_HOME=/usr/local/cuda setsid nohup "${CMD[@]}" > $D/server.log 2>&1 < /dev/null & SP=$!
ok=0; for i in $(seq 1 120); do [ "$(curl -s -o /dev/null -w %{http_code} -m 3 localhost:30000/health)" = 200 ] && { ok=1; break; }; kill -0 $SP 2>/dev/null || break; sleep 3; done
say "ready=$ok; gpu mem $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
curl -s localhost:30000/get_server_info 2>/dev/null | python3 -c "import json,sys;d=json.load(sys.stdin);print('version',d.get('version'),'lora_drain_wait_threshold',d.get('lora_drain_wait_threshold'))" >> $L 2>&1
kill $SP 2>/dev/null; sleep 8; pkill -9 -f "sglang.launch_server" 2>/dev/null; pkill -9 -f "sglang::" 2>/dev/null; sleep 3; say "stopped; gpu mem $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"; echo CHECK_DONE >> $L
