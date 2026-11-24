#!/usr/bin/env bash
# sourced by the run scripts: start_vllm / start_sglang / stop_engine, state(); flags as in followup/routing/PLAN.md (A2 notes) with prefix caching ON
A=/dev/shm/adapters_s0.02; MODEL=/dev/shm/models/Qwen2.5-1.5B-Instruct
state() { echo "load $(cut -d' ' -f1-3 /proc/loadavg) | extload $(/workspace/envs/vllm/bin/python /workspace/study/followup/kernel_rerun/extload.py check 6) | gpu $(nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw,utilization.gpu,memory.used --format=csv,noheader | tr -d ' ')"; }
start_vllm() {  # $1 = log, extra flags in $2...
  local log=$1; shift
  CMD=(/workspace/envs/vllm/bin/vllm serve $MODEL --served-model-name qwen --port 8000 --max-model-len 4096 --gpu-memory-utilization 0.85 --enable-lora --max-loras 16 --max-cpu-loras 64 --max-lora-rank 16 "$@" --lora-modules)
  for i in 0 1 2 3; do CMD+=("adapter_000$i=$A/adapter_000$i"); done
  "${CMD[@]}" > $log 2>&1 & SP=$!
  for i in $(seq 1 120); do [ "$(curl -s -o /dev/null -w %{http_code} -m 3 localhost:8000/health)" = 200 ] && return 0; kill -0 $SP 2>/dev/null || return 1; sleep 5; done; return 1; }
start_sglang() {
  local log=$1; shift
  export PATH=/workspace/envs/sglang/bin:$PATH CUDA_HOME=/usr/local/cuda
  CMD=(/workspace/envs/sglang/bin/python -m sglang.launch_server --model-path $MODEL --served-model-name qwen --port 30000 --mem-fraction-static 0.6 --context-length 4096
       --enable-lora --max-lora-rank 16 --max-loras-per-batch 16 --lora-target-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj "$@" --lora-paths)
  for i in 0 1 2 3; do CMD+=("adapter_000$i=$A/adapter_000$i"); done
  "${CMD[@]}" > $log 2>&1 & SP=$!
  for i in $(seq 1 120); do [ "$(curl -s -o /dev/null -w %{http_code} -m 3 localhost:30000/health)" = 200 ] && return 0; kill -0 $SP 2>/dev/null || return 1; sleep 5; done; return 1; }
stop_engine() { kill $SP 2>/dev/null; sleep 8; pkill -9 -f "EngineCore" 2>/dev/null; pkill -9 -f "sglang.launch_server" 2>/dev/null; pkill -9 -f "sglang::" 2>/dev/null; sleep 4; }
gpu_idle() { [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ]; }
