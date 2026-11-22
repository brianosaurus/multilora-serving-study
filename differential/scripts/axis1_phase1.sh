#!/bin/bash
# Axis 1 phase 1 for one model: noise floor (5 repeats + short prompts), C0 baseline, C7 concurrency, C6 prefix cache on/off.
# usage: axis1_phase1.sh MODELNAME MODEL_DIR ENGINES...   (MODEL_DIR on /dev/shm; reference files in $KVC/diff/ref/)
source /root/kvc/diff/harness/guard.sh; source /root/kvc/diff/harness/engine.sh; source $KVC/venv-ref/bin/activate
NAME=$1; MD=$2; shift 2; D=$KVC/diff; REF=$D/ref/${NAME}_raw_run1.main.jsonl; REFS=$D/ref/${NAME}_raw_run1.short.jsonl; REFH=$D/ref/${NAME}_hfdefault_run1.main.jsonl; CL="python $D/harness/tf_client.py"
meta(){ { date -u +%FT%TZ; echo "cmd: $(ps -o args= -p $PID | cut -c1-400)"; nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used --format=csv,noheader; } > $1; }
snap(){ curl -s localhost:$PORT/metrics > $1 2>/dev/null; }
for E in "$@"; do
  O=$D/raw/axis1/$NAME/$E; mkdir -p $O
  for CFG in default noprefix; do
    echo "######## $NAME $E $CFG"; start_engine $E $MD $CFG $O/server_$CFG.log || { stop_engine; echo "FAILED $E $CFG" >> $O/FAILED; continue; }
    meta $O/meta_$CFG.txt; echo "startup_secs $STARTUP_SECS" >> $O/meta_$CFG.txt
    if [ $CFG = default ]; then
      GEN=""; [ $E = mistralrs ] && GEN="--gen"
      for r in 1 2 3 4 5; do $CL $E $PORT $REF $O/nf_main_run$r.jsonl --tokdir $MD $GEN | tail -1; done
      $CL $E $PORT $REFS $O/nf_short.jsonl --tokdir $MD $GEN | tail -1
      $CL $E $PORT $REF $O/c7_conc32.jsonl --tokdir $MD --conc 32 $GEN | tail -1
      $CL $E $PORT $REF $O/c8_defaults.jsonl --tokdir $MD --gen --sampling defaults | tail -1   # C8: only temperature=0 in the request (what a user gets)
    fi
    snap $O/metrics_${CFG}_before.txt; $CL $E $PORT $REF $O/c6_${CFG}_pass1.jsonl --tokdir $MD --gen | tail -1; snap $O/metrics_${CFG}_mid.txt
    $CL $E $PORT $REF $O/c6_${CFG}_pass2.jsonl --tokdir $MD --gen | tail -1; snap $O/metrics_${CFG}_after.txt
    meta $O/meta_${CFG}_end.txt; stop_engine
  done
done; echo P1DONE
