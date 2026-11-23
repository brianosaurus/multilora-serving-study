#!/usr/bin/env bash
# Balanced kernel benchmark on a DEDICATED machine (plan: PLAN.md in this directory).
# Usage: run_rerun.sh --host EXPECTED_HOSTNAME --old OLD_BIN --new NEW_BIN --model DIR --adapters DIR --loadgen PATH --python PY --out DIR
#                    [--requests 150] [--warmup 20] [--blocks 6] [--max-load 2] [--dry-run]
# Refuses to start unless: hostname matches, the GPU is idle, and the 1-minute load average is <= max-load.
# Marks every run during which the load average rose above max-load. --dry-run only tests the script's logic (tiny counts, load check
# downgraded to a warning), and its output must be discarded or labeled as a dry run.
set -u
REQ=150; WARM=20; BLOCKS=6; MAXLOAD=2; DRY=0
while [ $# -gt 0 ]; do case $1 in
  --host) HOST=$2; shift 2;; --old) OLD=$2; shift 2;; --new) NEW=$2; shift 2;; --model) M=$2; shift 2;; --adapters) A=$2; shift 2;;
  --loadgen) LG=$2; shift 2;; --python) PY=$2; shift 2;; --out) OUT=$2; shift 2;; --requests) REQ=$2; shift 2;; --warmup) WARM=$2; shift 2;;
  --blocks) BLOCKS=$2; shift 2;; --max-load) MAXLOAD=$2; shift 2;; --dry-run) DRY=1; shift;; *) echo "unknown arg $1"; exit 2;; esac; done
for v in HOST OLD NEW M A LG PY OUT; do [ -n "${!v:-}" ] || { echo "missing --$(echo $v | tr A-Z a-z)"; exit 2; }; done
mkdir -p "$OUT"; LOG="$OUT/run.log"; SUM="$OUT/summary.txt"; : > "$LOG"; : > "$SUM"
say() { echo "$(date -u +%T) $*" | tee -a "$LOG"; }
load1() { cut -d' ' -f1 /proc/loadavg; }
over() { awk -v a="$1" -v b="$MAXLOAD" 'BEGIN{exit !(a>b)}'; }
[ "$DRY" = 1 ] && say "DRY RUN: counts are tiny and results must not be used"
# --- preconditions
[ "$(hostname)" = "$HOST" ] || { say "REFUSED: hostname is $(hostname), expected $HOST"; exit 3; }
nproc_=$(nproc); gpu_apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l); gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
{ [ "$gpu_apps" = 0 ] && [ "$gpu_mem" -lt 500 ]; } || { say "REFUSED: GPU is not idle ($gpu_apps compute processes, ${gpu_mem} MiB used)"; exit 4; }
l=$(load1)
if over "$l"; then
  if [ "$DRY" = 1 ]; then say "WARNING (dry run only): load average $l is above $MAXLOAD"; else say "REFUSED: 1-minute load average $l is above $MAXLOAD"; exit 5; fi
fi
say "host $(hostname) ok; $nproc_ cores; load $l; GPU idle; gpu: $(nvidia-smi --query-gpu=name,clocks.sm,temperature.gpu --format=csv,noheader)"
cp -f "$OLD" "$OUT/old2.bin"; chmod +x "$OUT/old2.bin"
declare -A BIN=( [old]="$OLD" [new]="$NEW" [old2]="$OUT/old2.bin" )
sha() { sha256sum "$1" | cut -c1-16; }
say "binaries: old $(sha $OLD)  new $(sha $NEW)  old2 $(sha $OUT/old2.bin)"
ORDERS=("old new old2" "new old2 old" "old2 old new" "new old old2" "old old2 new" "old2 new old")
names=$(for i in 0 1 2 3 4 5 6 7; do printf "adapter_%04d " $i; done)
gpu_state() { nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw,clocks_throttle_reasons.active --format=csv,noheader | tr -d ' '; }
run_cell() { # block pos bin name conc adapters
  blk=$1; pos=$2; bin=$3; name=$4; conc=$5; nad=$6
  ( while true; do cut -d' ' -f1 /proc/loadavg; sleep 2; done ) > "$OUT/.loadsamples" & sampler=$!
  g0=$(gpu_state); l0=$(load1)
  "$PY" "$LG" --engine mistralrs --url http://localhost:1234 --tokenizer "$M" --requests "$REQ" --warmup "$WARM" --seed 5 --concurrency "$conc" --adapters "$nad" \
        --popularity uniform --out "$OUT/raw/${name}_${bin}_b${blk}.json" > /dev/null 2>&1
  kill $sampler 2>/dev/null; wait $sampler 2>/dev/null
  g1=$(gpu_state); l1=$(load1); lmax=$(sort -rn "$OUT/.loadsamples" | head -1); flag=OK; over "${lmax:-0}" && flag=LOAD_ABOVE_LIMIT; over "$l0" && flag=LOAD_ABOVE_LIMIT
  echo "blk$blk pos$pos $bin $name load_start=$l0 load_end=$l1 load_max=${lmax:-NA} gpu_start=$g0 gpu_end=$g1 flag=$flag" >> "$SUM"
}
mkdir -p "$OUT/raw"
blk=0
for ord in "${ORDERS[@]:0:$BLOCKS}"; do
  blk=$((blk+1)); pos=0
  for bin in $ord; do
    pos=$((pos+1))
    over "$(load1)" && say "note: load $(load1) before server $bin (block $blk pos $pos)"
    CMD=("${BIN[$bin]}" serve -m "$M" -p 1234 --kv-cache 8GiB --max-seqs 256 --lora-max-adapters 64)
    for n in $names; do CMD+=(--lora "$n=$A/$n"); done
    setsid nohup "${CMD[@]}" > "$OUT/server_${bin}_b$blk.log" 2>&1 < /dev/null &
    SP=$!; for i in $(seq 1 120); do [ "$(curl -s -o /dev/null -w %{http_code} -m 3 localhost:1234/v1/models)" = 200 ] && break; sleep 3; done
    say "block $blk position $pos: $bin up (load $(load1))"
    "$PY" "$LG" --engine mistralrs --url http://localhost:1234 --tokenizer "$M" --requests "$REQ" --warmup "$WARM" --seed 5 --concurrency 8 --adapters 8 \
          --popularity uniform --out "$OUT/raw/warm_${bin}_b$blk.json" > /dev/null 2>&1
    for conc in 8 32; do for nad in 0 1 4 8; do run_cell $blk $pos $bin "c${conc}_a${nad}" $conc $nad; done; done
    kill $SP 2>/dev/null; sleep 5; kill -9 $SP 2>/dev/null; pkill -9 -f "${BIN[$bin]} serve" 2>/dev/null; sleep 4
  done
done
say "runs finished; flagged runs: $(grep -c LOAD_ABOVE_LIMIT "$SUM")"
"$PY" "$(dirname "$0")/analyze_rerun.py" "$OUT" "$BLOCKS" $([ "$DRY" = 1 ] && echo --dry-run) | tee "$OUT/analysis.txt" | tee -a "$LOG"
