#!/usr/bin/env bash
# preflight.sh: lists what the dedicated machine is missing for run_session.sh. Exit 0 only if nothing is missing. Run on that machine.
ROOT=${ROOT:-/workspace/study}; ENVS=${ENVS:-/workspace/envs}; BIN=${BIN:-/workspace/bin}; M=${M:-/dev/shm/models/Qwen2.5-1.5B-Instruct}
A05=${A05:-/dev/shm/adapters}; A02=${A02:-/dev/shm/adapters_s0.02_32}; miss=0
need() { [ -e "$1" ] || { echo "MISSING: $1  ($2)"; miss=1; }; }
need $ROOT/scripts/loadgen.py "repo"; need $ROOT/followup/kernel_rerun/run_rerun.sh "repo"; need $ROOT/followup/prefix_routing/results/raw/refs_prefix.json "3a references"
for e in vllm sglang sglang_main; do need $ENVS/$e/bin/python "environment $e"; done
need $M/config.json "model"; need $BIN/mistralrs_old "binary from Item D"; need $BIN/mistralrs_new "binary from Item D"
[ "$(ls $A05 2>/dev/null | grep -c '^adapter_')" -ge 128 ] || { echo "MISSING: 128 adapters in $A05"; miss=1; }
[ "$(ls $A02 2>/dev/null | grep -c '^adapter_')" -ge 32 ] || { echo "MISSING: 32 adapters (scale 0.02) in $A02"; miss=1; }
$ENVS/sglang_main/bin/python -c "import sglang;print('sglang_main', sglang.__version__)" 2>/dev/null || { echo "MISSING: sglang_main does not import"; miss=1; }
nvidia-smi --query-gpu=name --format=csv,noheader; [ $miss = 0 ] && echo "preflight ok"; exit $miss
