#!/bin/bash
source /root/kvc/diff/harness/guard.sh; source /root/kvc/diff/harness/engine.sh; source $KVC/venv-ref/bin/activate
N=Qwen2.5-1.5B-Instruct; MD=/dev/shm/kvc/models/$N; D=$KVC/diff; O=$D/raw/c8_endpoints; mkdir -p $O
for E in vllm sglang mistralrs; do echo "######## $E"; start_engine $E $MD default $O/server_$E.log || { stop_engine; continue; }
  python $D/harness/c8_endpoints.py $E $PORT $MD $D/ref/${N}_raw_run1.main.jsonl $D/ref/${N}_hfdefault_run1.main.jsonl $O/$E.json 2>&1 | tail -3; stop_engine; done; echo C8DONE
