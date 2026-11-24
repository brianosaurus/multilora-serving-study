#!/usr/bin/env bash
# Item B (plan: followup/pod_items/PLAN.md): GPU memory and KV capacity from startup logs at 16/32/64 slots (32 adapters) and 64 slots (128 adapters).
# vLLM 0.34.0 and SGLang 0.5.23. One server start per configuration; no request is sent.
D=/workspace/study/followup/pod_items/results/B; mkdir -p $D; L=$D/run.log; : > $L; : > $D/summary.tsv
M=/dev/shm/models/Qwen2.5-1.5B-Instruct; ENVS=/workspace/envs
say() { echo "$(date -u +%T) $*" >> $L; }
names() { for i in $(seq 0 $(( $1 - 1 ))); do printf "adapter_%04d " $i; done; }
for cfg in "vllm 16 32" "vllm 32 32" "vllm 64 32" "vllm 64 128" "sglang 16 32" "sglang 32 32" "sglang 64 32" "sglang 64 128"; do
  set -- $cfg; eng=$1; slots=$2; nad=$3; tag=${eng}_s${slots}_n${nad}; adir=/dev/shm/adapters; [ $nad = 128 ] && adir=/dev/shm/adapters_128
  [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { say "REFUSED $tag: GPU not idle"; continue; }
  say "start $tag; load $(cut -d' ' -f1-3 /proc/loadavg)"
  if [ $eng = vllm ]; then PORT=8000
    CMD=($ENVS/vllm/bin/vllm serve $M --served-model-name qwen --port 8000 --max-model-len 4096 --gpu-memory-utilization 0.85 --enable-lora --max-loras $slots --max-cpu-loras 128 --max-lora-rank 16 --lora-modules)
  else PORT=30000
    CMD=($ENVS/sglang/bin/python -m sglang.launch_server --model-path $M --served-model-name qwen --port 30000 --mem-fraction-static 0.85 --context-length 4096 --enable-lora --max-lora-rank 16 --max-loras-per-batch $slots --lora-target-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj --lora-paths)
  fi
  for n in $(names $nad); do CMD+=("$n=$adir/$n"); done
  PATH=$ENVS/$eng/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda setsid nohup "${CMD[@]}" > $D/server_$tag.log 2>&1 < /dev/null & SP=$!
  ok=0; for i in $(seq 1 200); do [ "$(curl -s -o /dev/null -w %{http_code} -m 3 localhost:$PORT/health)" = 200 ] && { ok=1; break; }; kill -0 $SP 2>/dev/null || break; sleep 3; done
  sleep 5; mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)
  say "$tag ready=$ok gpu_mem=$mem"; printf "%s\t%s\t%s\t%s\t%s\t%s\n" $tag $eng $slots $nad $ok "$mem" >> $D/summary.tsv
  kill $SP 2>/dev/null; sleep 8; pkill -9 -f "[v]llm serve|[s]glang.launch_server|[s]glang::|[V]LLM::EngineCore" 2>/dev/null; sleep 5
done
echo B_DONE >> $L
