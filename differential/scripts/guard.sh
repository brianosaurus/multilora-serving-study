# Source in every differential script: hostname, GPU idle, no foreign engine process. Exits non-zero on any failure.
source /root/kvc/scripts/common.sh   # exits on hostname mismatch; sets PATH/CUDA_HOME (SGLang scripts override CUDA_HOME with the venv's CUDA 13 shim)
gpu_idle(){ u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$u" -lt 500 ] || { echo "GPU NOT IDLE ($u MiB)"; return 1; }
  n=$(ps -eo args | grep -E "vllm serve|sglang.launch_server|mistralrs serve|python .*(vllm|sglang)" | grep -vE "grep|guard" | wc -l); [ "$n" -eq 0 ] || { echo "FOREIGN ENGINE PROCESS RUNNING ($n)"; return 1; }; }
