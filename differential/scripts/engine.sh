# Source after guard.sh. Functions: start_engine ENGINE MODEL_DIR CONFIG LOGFILE ; stop_engine
# ENGINE: vllm | sglang | mistralrs     CONFIG: default | nochunk | noprefix | custom (uses EXTRA_ARGS)
# Defaults arm = documented defaults + only what is needed to serve (see PLAN section 7). Context 8192 where the engine accepts it.
start_engine(){
  local E=$1 MD=$2 CFG=$3 LOG=$4; local EXTRA="$EXTRA_ARGS"; PORT=; PID=
  gpu_idle || return 1
  case $E in
   vllm)
     source $KVC/venv-vllm/bin/activate; PORT=8000
     case $CFG in nochunk) EXTRA="--no-enable-chunked-prefill $EXTRA";; noprefix) EXTRA="--no-enable-prefix-caching $EXTRA";; esac
     setsid vllm serve $MD --served-model-name m --port $PORT --max-model-len 8192 --max-num-seqs 32 --gpu-memory-utilization ${GPU_UTIL:-0.85} --enable-prompt-tokens-details $EXTRA > $LOG 2>&1 & PID=$! ;;
   sglang)
     source $KVC/venv-sglang/bin/activate; PORT=30000
     # default SGLang runs use the system toolchain (CUDA 12.8 nvcc) like the multi-LoRA study; the CUDA 13 shim is only needed for HiCache (see kv-conformance B5) and breaks flashinfer's JIT
     case $CFG in nochunk) EXTRA="--chunked-prefill-size -1 $EXTRA";; noprefix) EXTRA="--disable-radix-cache $EXTRA";; esac
     setsid python -m sglang.launch_server --model-path $MD --served-model-name m --port $PORT --context-length 8192 --max-running-requests 32 --mem-fraction-static ${MEM_FRAC:-0.85} --enable-cache-report --enable-metrics $EXTRA > $LOG 2>&1 & PID=$! ;;
   mistralrs)
     PORT=18001; export LD_LIBRARY_PATH=$KVC/mrs/lib:${LD_LIBRARY_PATH:-}
     case $CFG in nochunk) EXTRA="--max-prefill-chunk-tokens 4096 $EXTRA";; noprefix) EXTRA="--prefix-cache-n 0 $EXTRA";; default) EXTRA="--max-prefill-chunk-tokens 512 $EXTRA";; esac
     setsid $KVC/mrs/mistralrs serve -m $MD --port $PORT --max-seqs 32 --kv-cache ${MRS_KV:-6GiB} --seed 0 $MRS_CTX $EXTRA > $LOG 2>&1 & PID=$! ;;
  esac
  START_T0=$(date +%s.%N)
  for i in $(seq 1 240); do
    if [ $E = mistralrs ]; then curl -s localhost:$PORT/v1/completions -H 'Content-Type: application/json' -d '{"model":"default","prompt":"hi","max_tokens":1,"temperature":0}' 2>/dev/null | grep -q '"choices"' && break
    else curl -sf localhost:$PORT/v1/models >/dev/null 2>&1 && break; fi
    kill -0 $PID 2>/dev/null || { echo "SERVER DIED AT START"; sed "s/\x1b\[[0-9;]*m//g" $LOG | grep -E "Error|error" | tail -3 | cut -c1-250; return 1; }; sleep 2; done
  STARTUP_SECS=$(python3 -c "import time;print(round(time.time()-$START_T0,1))"); echo "started $E $CFG in ${STARTUP_SECS}s"
}
stop_engine(){ [ -n "$PID" ] && { kill -TERM -- -$PID 2>/dev/null; sleep 8; kill -KILL -- -$PID 2>/dev/null; }
  pkill -TERM -f "$KVC/mrs/mistralrs serve" 2>/dev/null; sleep 2
  for i in $(seq 1 40); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$u" -lt 500 ] && break; sleep 2; done; }
