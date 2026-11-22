#!/bin/bash
# usage: run_ref.sh NAME MODEL_DIR   (MODEL_DIR must be on /dev/shm). Two runs: run1 and run2 (reference's own repeat spread).
source "$(dirname "$0")/guard.sh"; source $KVC/venv-ref/bin/activate
NAME=$1; MD=$2; VAR=${3:-raw}; D=$KVC/diff; mkdir -p $D/ref
for r in 1 2; do gpu_idle || exit 2; python $D/harness/reference.py $MD $NAME $D/ref/${NAME}_${VAR}_run$r $D/prompts $VAR 2>&1 | tail -2; done
