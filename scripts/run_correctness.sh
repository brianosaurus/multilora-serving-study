#!/usr/bin/env bash
# Gate 2: start each engine with 4 adapters, run scripts/check_engine.py, stop it.
# One engine at a time on the GPU. The engine launch lines below are the per-engine
# record of how each one is configured for the correctness check.
set -u
PY=/root/study/venvs/tools/bin/python
M=/dev/shm/models/Qwen2.5-1.5B-Instruct
A=/dev/shm/adapters
REF=/workspace/study/calib/ref_0.05.json
OUT=/workspace/study/correctness
LOGS=/root/study/logs
SC=/root/study/scripts
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
mkdir -p $OUT
NAMES="adapter_0000 adapter_0001 adapter_0002 adapter_0003"
TARGETS="q_proj k_proj v_proj o_proj gate_proj up_proj down_proj"

gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader; }
wait_ready() { # url pid seconds
  for i in $(seq 1 $(( $3 / 3 ))); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' -m 3 "$1")" = 200 ] && return 0
    kill -0 $2 2>/dev/null || return 1
    sleep 3
  done
  return 1
}
run_one() { # engine port healthpath ; remaining args = server command
  eng=$1; port=$2; health=$3; shift 3
  echo "=== $eng: starting $(date -u +%T), GPU before: $(gpu_used)"
  nohup setsid "$@" > $LOGS/correct_$eng.log 2>&1 < /dev/null &
  pid=$!
  t0=$(date +%s)
  if wait_ready "http://localhost:$port$health" $pid 420; then
    echo "  ready after $(( $(date +%s) - t0 )) s, GPU used: $(gpu_used)"
    $PY $SC/check_engine.py --engine $eng --url http://localhost:$port --reference $REF \
      --tokenizer $M --out $OUT/$eng.json 2>&1 | tail -6
  else
    echo "  NOT READY; last log lines:"; tail -8 $LOGS/correct_$eng.log | cut -c1-200
  fi
  kill $pid 2>/dev/null; sleep 6
  pkill -9 -f "[v]llm serve|[l]aunch_server|[r]elease/mistralrs" 2>/dev/null; sleep 3
  echo "  after shutdown, GPU used: $(gpu_used)"
}

only=${1:-all}
if [ "$only" = all ] || [ "$only" = vllm ]; then
  mods=""; for n in $NAMES; do mods="$mods $n=$A/$n"; done
  run_one vllm 8000 /health /root/study/venvs/vllm/bin/vllm serve $M --served-model-name qwen \
    --port 8000 --max-model-len 4096 --gpu-memory-utilization 0.85 \
    --enable-lora --max-loras 4 --max-lora-rank 16 --lora-modules $mods
fi
if [ "$only" = all ] || [ "$only" = sglang ]; then
  paths=""; for n in $NAMES; do paths="$paths $n=$A/$n"; done
  PATH=/root/study/venvs/sglang/bin:$PATH run_one sglang 30000 /health \
    /root/study/venvs/sglang/bin/python -m sglang.launch_server --model-path $M --served-model-name qwen \
    --port 30000 --mem-fraction-static 0.85 --context-length 4096 \
    --enable-lora --max-lora-rank 16 --max-loras-per-batch 4 --lora-target-modules $TARGETS --lora-paths $paths
fi
if [ "$only" = all ] || [ "$only" = mistralrs ]; then
  loras=""; for n in $NAMES; do loras="$loras --lora $n=$A/$n"; done
  run_one mistralrs 1234 /v1/models /workspace/build/target-mrs/release/mistralrs serve -m $M -p 1234 \
    --kv-cache 8GiB $loras
fi
echo CHECK_DONE
