#!/usr/bin/env bash
# One session on the dedicated machine (plan: followup/dedicated_session/PLAN.md). Blocks, in this order:
#   B1  3a repeat: SGLang 0.5.23 radix-on passes (clears the 3a flag)
#   B2  4a: kernel rerun (followup/kernel_rerun/run_rerun.sh, with its own checks; Amendment 2: shuffled cell order)
#   B3  4b: cliff rerun with the 0.02 adapters, vLLM and SGLang 0.5.23, 16 and 32 slots, Zipf and uniform
#   B4  3b: SGLang v0.5.23 against main (drainer), defaults and --lora-drain-wait-threshold 1.0, 16 slots, plus 32 slots at Zipf 40
#   B5  3c: 128 adapters on 64 slots, Zipf, vLLM and SGLang 0.5.23, 20 and 40 rps
# Checks (those of run_rerun.sh): hostname equals --host; GPU idle; 1-minute load average <= 2 once before the first run; external CPU
# (extload.py) <= 1 core over 10 s before the first run and over 6 s before every run (waits up to 120 s, then runs and flags); any sample
# above 2 external cores during a run flags it (LOAD_ABOVE_LIMIT). Reference runs at the start and end of every server session; drift
# above 5% marks the session DRIFT. Output length: every measured request must report 128 chunks, else flag LENGTH. Flagged runs are
# reported, never silently used (analyze_session.py lists them). --dry-run: tiny counts, checks are warnings, output must be discarded.
# Usage: run_session.sh --host HOSTNAME --out DIR [--seed N] [--blocks "B1 B2 B3 B4 B5"] [--dry-run]
set -u
HOST=; OUT=; SEED=20261124; BLOCKS="B1 B2 B3 B4 B5"; DRY=0
ROOT=${ROOT:-/workspace/study}; ENVS=${ENVS:-/workspace/envs}; BIN=${BIN:-/workspace/bin}
M=${M:-/dev/shm/models/Qwen2.5-1.5B-Instruct}; A05=${A05:-/dev/shm/adapters}; A02=${A02:-/dev/shm/adapters_s0.02_32}
while [ $# -gt 0 ]; do case $1 in --host) HOST=$2; shift 2;; --out) OUT=$2; shift 2;; --seed) SEED=$2; shift 2;;
  --blocks) BLOCKS=$2; shift 2;; --dry-run) DRY=1; shift;; *) echo "unknown arg $1"; exit 2;; esac; done
[ -n "$HOST" ] && [ -n "$OUT" ] || { echo "need --host and --out"; exit 2; }
HERE="$ROOT/followup/dedicated_session"; SC="$ROOT/scripts"; PYL=$ENVS/vllm/bin/python; EXT="$PYL $ROOT/followup/kernel_rerun/extload.py"
NREQ=500; WARM=50; REPS=3; [ "$DRY" = 1 ] && { NREQ=30; WARM=5; REPS=1; }
mkdir -p "$OUT/raw"; LOG="$OUT/session.log"; TSV="$OUT/cells.tsv"; : > "$LOG"; : > "$TSV"
say() { echo "$(date -u +%T) $*" | tee -a "$LOG"; }
gt() { awk -v a="$1" -v b="$2" 'BEGIN{exit !(a>b)}'; }
gpu_idle() { [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] && [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 500 ]; }
hyg() { nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw,clocks_throttle_reasons.active --format=csv,noheader | tr -d ' '; }
[ "$DRY" = 1 ] && say "DRY RUN: tiny counts, results must be discarded"
[ "$(hostname)" = "$HOST" ] || { say "REFUSED: hostname is $(hostname), expected $HOST"; exit 3; }
gpu_idle || { say "REFUSED: GPU not idle"; exit 4; }
read -r em ex l0 <<< "$($EXT check 10)"
if gt "$l0" 2 || gt "$em" 1; then
  if [ "$DRY" = 1 ]; then say "WARNING (dry run): load $l0 or external CPU median $em above limit"; else say "REFUSED: load average $l0 (limit 2) or external CPU median $em cores over 10 s (limit 1)"; exit 5; fi
fi
say "host $(hostname) ok; $(nproc) cores; load $l0; external CPU median $em max $ex; GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader); seed $SEED; blocks: $BLOCKS"
{ echo "repo commit: $(cat $ROOT/SOURCE_COMMIT.txt 2>/dev/null)"; for e in vllm sglang sglang_main; do echo "== $e"; uv pip list --python $ENVS/$e/bin/python 2>/dev/null | grep -i -E "^(vllm|sglang|sglang-kernel|torch|transformers|flashinfer-python) "; done; \
  (cd $ROOT/../src/sglang-main 2>/dev/null && echo "sglang main commit: $(git rev-parse HEAD)"); sha256sum $BIN/* 2>/dev/null; } > "$OUT/versions.txt" 2>&1

wait_quiet() { pre=NA; for t in $(seq 1 20); do read -r pm px pl <<< "$($EXT check 6)"; pre=$pm; gt "$pm" 1 || break; sleep 4; done; echo "$pre"; }
names() { for i in $(seq 0 $(( $1 - 1 ))); do printf "adapter_%04d " "$i"; done; }
PID=
start_server() { # kind slots nad adapterdir extra...   kind: vllm | sglang | sglang_main
  kind=$1; slots=$2; nad=$3; adir=$4; shift 4; extra=("$@")
  if [ $kind = vllm ]; then PORT=8000; EP=vllm
    CMD=($ENVS/vllm/bin/vllm serve $M --served-model-name qwen --port 8000 --max-model-len 4096 --gpu-memory-utilization 0.85 --enable-lora --max-loras $slots --max-cpu-loras $(( nad > 64 ? nad : 64 )) --max-lora-rank 16 "${extra[@]}" --lora-modules)
  else PORT=30000; EP=sglang
    CMD=($ENVS/$kind/bin/python -m sglang.launch_server --model-path $M --served-model-name qwen --port 30000 --mem-fraction-static 0.85 --context-length 4096 --enable-lora --max-lora-rank 16 --max-loras-per-batch $slots --lora-target-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj "${extra[@]}" --lora-paths)
  fi
  for n in $(names $nad); do CMD+=("$n=$adir/$n"); done
  if [ $kind = vllm ]; then ENVSET=(env PATH=$ENVS/vllm/bin:$PATH); else ENVSET=(env PATH=$ENVS/$kind/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda); fi   # vLLM without CUDA_HOME (flashinfer JIT with nvcc 12.8 against torch cu130 crashed vLLM on the pod)
  setsid nohup "${ENVSET[@]}" "${CMD[@]}" > "$OUT/server_$TAG.log" 2>&1 < /dev/null &
  PID=$!; for i in $(seq 1 200); do [ "$(curl -s -o /dev/null -w %{http_code} -m 3 localhost:$PORT/health)" = 200 ] && return 0; kill -0 $PID 2>/dev/null || return 1; sleep 3; done; return 1; }
stop_server() { kill $PID 2>/dev/null; sleep 6; pkill -9 -f "[v]llm serve|[s]glang.launch_server|[s]glang::|[V]LLM::EngineCore" 2>/dev/null; sleep 4; }
loadgen() { # out rate nreq warm adapters offset popularity seed
  $PYL $SC/loadgen.py --engine $EP --url http://localhost:$PORT --tokenizer $M --rate $2 --requests $3 --warmup $4 --adapters $5 --adapter-offset 0 --popularity $7 --seed $8 --out "$1" > /dev/null 2>&1; }
run_cell() { # tag workload rate rep nad pop
  tag=$1; wl=$2; rate=$3; rep=$4; nad=$5; pop=$6; f="$OUT/raw/${tag}_${wl}_r${rate}_rep${rep}.json"
  pre=$(wait_quiet); rm -f "$OUT/.samples"; $PYL $ROOT/followup/kernel_rerun/extload.py sample "$OUT/.samples" 2 & sm=$!; g0=$(hyg)
  loadgen "$f" $rate $NREQ $WARM $nad 0 $pop $((1+rep)); kill $sm 2>/dev/null; wait $sm 2>/dev/null
  emax=$(awk 'BEGIN{m=0}{if($5>m)m=$5}END{print m}' "$OUT/.samples" 2>/dev/null); flag=OK
  gt "${emax:-0}" 2 && flag=LOAD_ABOVE_LIMIT; gt "$pre" 1 && flag=LOAD_ABOVE_LIMIT
  info=$(python3 $HERE/cellinfo.py "$f" 2>/dev/null || echo "NA	NA	NA	NA	NA	NA	NA"); [ "$(echo "$info" | cut -f6)" = 0 ] || flag="$flag+LENGTH"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\text_before=%s ext_max=%s gpu=%s>%s\n" "$BLOCK" "$tag" "$wl" "$rate" "$rep" "$flag" "$info" "$pre" "$emax" "$g0" "$(hyg)" >> "$TSV"; }
server_session() { # tag kind slots nad adir "extra flags" "workload:pop:nad:rates ..." 
  TAG=$1; kind=$2; slots=$3; nad=$4; adir=$5; extra=($6); specs=$7
  say "session $TAG ($kind, $slots slots, $nad adapters, extra: ${extra[*]:-none}) start; GPU $(hyg)"
  gpu_idle || { say "  REFUSED: GPU not idle"; return 1; }
  start_server $kind $slots $nad $adir "${extra[@]}" || { say "  server did not start; tail:"; tail -n 6 "$OUT/server_$TAG.log" | tee -a "$LOG"; stop_server; echo -e "$BLOCK\t$TAG\tSTART_FAILED" >> "$TSV"; return 1; }
  loadgen "$OUT/raw/${TAG}_warm1.json" 10 150 0 0 0 uniform 998; loadgen "$OUT/raw/${TAG}_warm2.json" 40 400 0 0 0 uniform 997
  loadgen "$OUT/raw/${TAG}_ref_start.json" 20 200 50 0 0 uniform 999
  for spec in $specs; do IFS=: read -r wl pop nadw rates <<< "$spec"
    for rate in ${rates//,/ }; do [ "$DRY" = 1 ] && [ "$rate" != "${rates%%,*}" ] && continue; for rep in $(seq 0 $((REPS-1))); do run_cell $TAG $wl $rate $rep $nadw $pop; done; done; done
  loadgen "$OUT/raw/${TAG}_ref_end.json" 20 200 50 0 0 uniform 999
  dr=$(python3 - "$OUT/raw/${TAG}_ref_start.json" "$OUT/raw/${TAG}_ref_end.json" <<'PY'
import json, sys
a, b = (json.load(open(p))["summary"]["e2e_ms"]["p50"] for p in sys.argv[1:3]); d = (b - a) / a * 100
print(f"{d:+.1f}% {'DRIFT' if abs(d) > 5 else 'ok'}")
PY
)
  say "  reference drift $TAG: $dr"; echo -e "$BLOCK\t$TAG\tDRIFT_CHECK\t$dr" >> "$TSV"; stop_server; say "  stopped; GPU used $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"; }
shuf_seeded() { python3 -c "import random,sys;l=sys.argv[2:];random.Random(int(sys.argv[1])).shuffle(l);print(' '.join(l))" "$@"; }

for BLOCK in $BLOCKS; do say "=== block $BLOCK"; case $BLOCK in
B1) # 3a repeat of the SGLang radix-on passes
  cd $ROOT/followup/prefix_routing; pre=$(wait_quiet); rm -f "$OUT/.samples"; $PYL $ROOT/followup/kernel_rerun/extload.py sample "$OUT/.samples" 2 & sm=$!
  PASS_TAG=passes_quiet_repeat A_DIR=$A02 MODEL=$M ./run_passes.sh sglang; kill $sm 2>/dev/null; wait $sm 2>/dev/null
  emax=$(awk 'BEGIN{m=0}{if($5>m)m=$5}END{print m}' "$OUT/.samples" 2>/dev/null); flag=OK; { gt "${emax:-0}" 2 || gt "$pre" 1; } && flag=LOAD_ABOVE_LIMIT
  PASS_TAG=passes_quiet_repeat $PYL $ROOT/followup/prefix_routing/analyze_passes.py > "$OUT/b1_analysis.txt" 2>&1
  echo -e "B1\t3a_repeat\tpasses\t-\t-\t$flag\text_before=$pre ext_max=$emax" >> "$TSV"; say "B1 done, flag $flag"; cd - >/dev/null;;
B2) # 4a
  bash $ROOT/followup/kernel_rerun/run_rerun.sh --host $HOST --old $BIN/mistralrs_old --new $BIN/mistralrs_new --model $M --adapters $A05 \
     --loadgen $SC/loadgen.py --python $PYL --out $OUT/kernel_rerun --seed $SEED $([ "$DRY" = 1 ] && echo "--dry-run");;
B3) # 4b: 0.02 adapters, 32 of them
  for s in $(shuf_seeded $SEED vllm:16 vllm:32 sglang:16 sglang:32); do eng=${s%%:*}; sl=${s##*:}
    server_session "4b_${eng}_s${sl}" $eng $sl 32 $A02 "" "zipf32:zipf:32:10,20,40 uni32:uniform:32:10,20,40"; done;;
B4) # 3b: SGLang 0.5.23 against main, study adapters (scale 0.05), 16 slots
  for s in $(shuf_seeded $SEED sglang:def sglang:drain1 sglang_main:def sglang_main:drain1); do k=${s%%:*}; v=${s##*:}; ex=""; [ $v = drain1 ] && ex="--lora-drain-wait-threshold 1.0"
    server_session "3b_${k}_${v}_s16" $k 16 32 $A05 "$ex" "zipf32:zipf:32:20,40,80 uni32:uniform:32:20,40"; done
  for k in $(shuf_seeded $SEED sglang sglang_main); do server_session "3b_${k}_def_s32" $k 32 32 $A05 "" "zipf32:zipf:32:40"; done;;
B5) # 3c: 128 adapters, 64 slots
  for eng in $(shuf_seeded $SEED vllm sglang); do server_session "3c_${eng}_s64_n128" $eng 64 128 $A05 "" "zipf128:zipf:128:20,40"; done;;
esac; done
say "session finished; cells flagged: $(grep -c -E 'LOAD_ABOVE_LIMIT|LENGTH' $TSV)"
python3 $HERE/analyze_session.py "$OUT" | tee "$OUT/analysis.txt" | tee -a "$LOG"
