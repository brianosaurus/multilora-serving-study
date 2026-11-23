#!/usr/bin/env bash
# Calibration sweep: generate seeds 0-3 at each B scale, run refs.py and analyze.py (plan: ../PLAN.md)
cd /root/k/routing; PY=/root/k/venvgpu/bin/python; L=sweep.log; : > $L
for s in 0.02 0.01 0.1; do
  echo "$(date -u +%T) scale $s: generating adapters" >> $L
  $PY /root/k/scripts/make_adapters.py --base /root/k/model --out /root/k/routing/adapters_s$s --count 4 --first-seed 0 --b-std $s >> sweep_gen_$s.log 2>&1
  echo "$(date -u +%T) scale $s: adapters done ($(ls adapters_s$s | grep -c adapter_)), sha256 of adapter_0000: $(sha256sum adapters_s$s/adapter_0000/adapter_model.safetensors 2>/dev/null | cut -c1-16)" >> $L
  ADAPTER_DIR=/root/k/routing/adapters_s$s $PY refs.py prompts.json refs_s$s.json 8 > refs_s$s.log 2>&1
  echo "$(date -u +%T) scale $s: references done" >> $L
  $PY analyze.py refs_s$s.json > analyze_s$s.txt 2>&1
  echo "$(date -u +%T) scale $s: analysis done: $(tail -1 analyze_s$s.txt | cut -c1-150)" >> $L
done
echo SWEEP_DONE >> $L
