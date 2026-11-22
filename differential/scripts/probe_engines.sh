#!/bin/bash
# Probe each engine's teacher-forcing response format on 2 Qwen prompts. One engine at a time.
source /root/kvc/diff/harness/guard.sh; source /root/kvc/diff/harness/engine.sh
MD=/dev/shm/kvc/models/Qwen2.5-1.5B-Instruct; REF=$KVC/diff/ref/Qwen2.5-1.5B-Instruct_run1.main.jsonl; mkdir -p $KVC/diff/raw/probe
for E in ${ENGINES:-vllm sglang mistralrs}; do
  echo "######## $E"; start_engine $E $MD default $KVC/diff/raw/probe/server_$E.log || { stop_engine; continue; }
  source $KVC/venv-ref/bin/activate; python $KVC/diff/harness/tf_client.py $E $PORT $REF /dev/null --probe --tokdir $MD 2>&1 | tail -6 | cut -c1-950
  stop_engine
done; echo PROBEDONE
