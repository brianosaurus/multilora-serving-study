#!/usr/bin/env bash
# Phase 2 runner (plan: followup/sglang_cliff/PLAN.md, "Phase 2"). Usage: HOST_EXPECT=48bd787e389e ./run_phase2.sh A|B|C
R=/root/msf_study; D=$R/followup/sglang_cliff/results/phase2; mkdir -p $D; L=$D/run.log; PART=${1:?part A|B|C}
say() { echo "$(date -u +%T) $*" | tee -a $L; }
HOST_EXPECT=${HOST_EXPECT:-48bd787e389e}
[ "$(hostname)" = "$HOST_EXPECT" ] || { echo "REFUSED: host $(hostname) is not $HOST_EXPECT"; exit 2; }
M=/dev/shm/models/Qwen2.5-1.5B-Instruct; A=/dev/shm/adapters; PYL=${PYL:-/root/envs/sglang/bin/python}; TR=$R/followup/sglang_cliff/trace
busy() { [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" != 0 ] || pgrep -f "[r]un_rerun.sh|[r]un_session.sh|[r]un_phase1.sh" >/dev/null; }
declare -A ENVOF=( [v23]=/root/envs/sglang [main]=/root/envs/sglang_main )
SP=""
start_sglang() { # tag eng slots trace(0|1)
  local tag=$1 eng=$2 slots=$3 trace=$4 O=$D/$1; mkdir -p $O; E=${ENVOF[$eng]}
  busy && { say "REFUSED $tag: GPU or another job busy"; return 1; }
  say "start $tag host $(hostname) load $(cut -d' ' -f1-3 /proc/loadavg) external CPU: $($PYL $R/followup/kernel_rerun/extload.py check 10 2>&1 | tail -1)"
  if [ "${NOLORA:-0}" = 1 ]; then CMD=($E/bin/python -m sglang.launch_server --model-path $M --served-model-name qwen --port 30000 --mem-fraction-static 0.85 --context-length 4096 $EXTRA)
  else CMD=($E/bin/python -m sglang.launch_server --model-path $M --served-model-name qwen --port 30000 --mem-fraction-static 0.85 --context-length 4096 --enable-lora --max-lora-rank 16 --max-loras-per-batch $slots --lora-target-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj $EXTRA --lora-paths)
  for i in $(seq 0 31); do CMD+=("adapter_$(printf %04d $i)=$A/adapter_$(printf %04d $i)"); done; fi
  if [ $trace = 1 ]; then PP="PYTHONPATH=$TR${PATCHPY:+:$PATCHPY} SGLANG_CLIFF_TRACE_DIR=$O/trace $XENV"; else PP="SGLANG_CLIFF_NOTRACE=1"; fi
  setsid nohup env PATH=$E/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda $PP "${CMD[@]}" > $O/server.log 2>&1 < /dev/null & SP=$!
  ok=0; for i in $(seq 1 600); do [ "$(curl -s -o /dev/null -w %{http_code} -m 3 localhost:30000/health)" = 200 ] && { ok=1; break; }; kill -0 $SP 2>/dev/null || break; sleep 3; done
  say "$tag ready=$ok trace=$trace"; [ $ok = 1 ]
}
stop_server() { kill $SP 2>/dev/null; sleep 8; pkill -9 -f "[s]glang.launch_server|[s]glang::|[v]llm serve|[V]LLM::EngineCore" 2>/dev/null; sleep 5; }
load() { # tag engine url rate seed
  local O=$D/$1; echo "{\"rate\":$4,\"t_start\":$(date +%s.%N)," >> $O/marks.tmp
  $PYL $R/scripts/loadgen.py --engine $2 --url $3 --tokenizer $M --rate $4 --requests 500 --warmup 50 --adapters 32 --adapter-offset 0 --popularity zipf --seed $5 ${RIDP:+--rid-prefix $RIDP-$4} --out $O/zipf32_r$4.json | cut -c1-300 | tee -a $L
  echo "\"t_end\":$(date +%s.%N)}" >> $O/marks.tmp; tr -d '\n' < $O/marks.tmp >> $O/marks.jsonl; echo >> $O/marks.jsonl; rm -f $O/marks.tmp
}
warm() { for s in 998 997; do $PYL $R/scripts/loadgen.py --engine $1 --url $2 --tokenizer $M --rate 40 --requests 300 --warmup 0 --adapters 0 --seed $s --out $D/$3/warm_$s.json >/dev/null 2>&1; done; }
case $PART in
A) # tracing overhead: main, 16 slots, 20 rps; U P P U U P
  n=0; for v in ${ORDER:-U P P U U P}; do n=$((n+1)); tag=A_${n}_$v; tr=0; [ $v = P ] && tr=1
    start_sglang $tag main 16 $tr && { warm sglang http://localhost:30000 $tag; load $tag sglang http://localhost:30000 20 1; [ $tr = 1 ] && say "$tag trace lines: $(cat $D/$tag/trace/*.jsonl 2>/dev/null | wc -l)"; }
    stop_server; say "done $tag"; done;;
B) for cfg in v23-16 main-16 main-32 v23-32; do eng=${cfg%%-*}; sl=${cfg#*-}; tag=B_$cfg
    start_sglang $tag $eng $sl 1 && { warm sglang http://localhost:30000 $tag; for rate in 20 40; do load $tag sglang http://localhost:30000 $rate 1; done; say "$tag trace lines: $(cat $D/$tag/trace/*.jsonl | wc -l)"; }
    stop_server; say "done $tag"; done;;
C) tag=C_vllm-16; O=$D/$tag; mkdir -p $O; E=/root/envs/vllm
  busy && { say "REFUSED $tag: busy"; exit 1; }
  say "start $tag host $(hostname) load $(cut -d' ' -f1-3 /proc/loadavg) external CPU: $($PYL $R/followup/kernel_rerun/extload.py check 10 2>&1 | tail -1)"
  CMD=($E/bin/vllm serve $M --served-model-name qwen --port 8000 --max-model-len 4096 --gpu-memory-utilization 0.85 --enable-lora --max-loras 16 --max-cpu-loras 64 --max-lora-rank 16 --lora-modules)
  for i in $(seq 0 31); do CMD+=("adapter_$(printf %04d $i)=$A/adapter_$(printf %04d $i)"); done
  setsid nohup env PATH=$E/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda "${CMD[@]}" > $O/server.log 2>&1 < /dev/null & SP=$!
  ok=0; for i in $(seq 1 900); do [ "$(curl -s -o /dev/null -w %{http_code} -m 3 localhost:8000/health)" = 200 ] && { ok=1; break; }; kill -0 $SP 2>/dev/null || break; sleep 3; done
  say "$tag ready=$ok"
  if [ $ok = 1 ]; then warm vllm http://localhost:8000 $tag
    for rate in 20 40; do curl -s localhost:8000/metrics > $O/metrics_before_r$rate.txt; load $tag vllm http://localhost:8000 $rate 1; curl -s localhost:8000/metrics > $O/metrics_after_r$rate.txt; done; fi
  stop_server; say "done $tag";;
D) # diagnostic (Phase 3 step 1): main, 16 slots, drainer on, traced; EXTRA flags per cell
  for cfg in "d1:--lora-drain-wait-threshold 1.0" "d02:--lora-drain-wait-threshold 0.2"; do tag=D_main-16-${cfg%%:*}; EXTRA="${cfg#*:}"
    start_sglang $tag main 16 1 && { warm sglang http://localhost:30000 $tag; for rate in 20 40; do load $tag sglang http://localhost:30000 $rate 1; done; say "$tag trace lines: $(cat $D/$tag/trace/*.jsonl | wc -l)"; }
    stop_server; say "done $tag"; done;;
E) # Phase 3 A/B: main 16 slots traced; U = stock main, P = depth-first patch (worktree on PYTHONPATH + flag); order U P P U U P
  PATCHDIR=/root/src/sglang-lora-depth/python
  n=0; for v in ${ORDER:-U P P U U P}; do n=$((n+1)); tag=E_main-16-$v$n; PATCHPY=; XENV=
    [ $v = P ] && { PATCHPY=$PATCHDIR; XENV="SGLANG_LORA_DEPTH_FIRST=1"; }
    start_sglang $tag main 16 1 && { [ $v = P ] && say "$tag sglang from: $(PYTHONPATH=$PATCHPY $E/bin/python -c 'import sglang;print(sglang.__file__)')"; warm sglang http://localhost:30000 $tag; for rate in 20 40; do load $tag sglang http://localhost:30000 $rate 1; done; say "$tag trace lines: $(cat $D/$tag/trace/*.jsonl | wc -l)"; }
    stop_server; say "done $tag"; done;;
G) # Phase 3 option 3: slot sweep on stock main, traced
  for sl in 20 24 28; do tag=G_main-$sl; PATCHPY=; XENV=
    start_sglang $tag main $sl 1 && { warm sglang http://localhost:30000 $tag; for rate in 20 40; do load $tag sglang http://localhost:30000 $rate 1; done; say "$tag trace lines: $(cat $D/$tag/trace/*.jsonl | wc -l)"; }
    stop_server; say "done $tag"; done;;
H) # Phase 3 option 5: decode CUDA graph limit raised, stock main code
  for n in 1 2 3; do tag=H_main-16-cg256-$n; PATCHPY=; XENV=; EXTRA="--cuda-graph-max-bs-decode 256"
    start_sglang $tag main 16 1 && { warm sglang http://localhost:30000 $tag; for rate in 20 40; do load $tag sglang http://localhost:30000 $rate 1; done; say "$tag trace lines: $(cat $D/$tag/trace/*.jsonl | wc -l)"; }
    stop_server; say "done $tag"; done;;
I) # step-time microbench: closed loop at fixed concurrency, base model vs 16 uniform adapters; three servers
  micro() { # tag engine url
    local O=$D/$1; for N in 16 48 96; do
      $PYL $R/scripts/loadgen.py --engine $2 --url $3 --tokenizer $M --concurrency $N --requests 300 --warmup 50 --adapters 0 --seed 5 --out $O/micro_N${N}_base.json | cut -c1-220 | tee -a $L
      $PYL $R/scripts/loadgen.py --engine $2 --url $3 --tokenizer $M --concurrency $N --requests 300 --warmup 50 --adapters 16 --popularity uniform --seed 5 --out $O/micro_N${N}_lora16.json | cut -c1-220 | tee -a $L
    done; }
  for cfg in "I_main-16-stock:" "I_main-16-cg256:--cuda-graph-max-bs-decode 256"; do tag=${cfg%%:*}; EXTRA="${cfg#*:}"; PATCHPY=; XENV=
    start_sglang $tag main 16 0 && { warm sglang http://localhost:30000 $tag; micro $tag sglang http://localhost:30000; }
    stop_server; say "done $tag"; done
  tag=I_vllm-16; O=$D/$tag; mkdir -p $O; E=/root/envs/vllm
  busy && { say "REFUSED $tag: busy"; exit 1; }
  say "start $tag host $(hostname) load $(cut -d' ' -f1-3 /proc/loadavg) external CPU: $($PYL $R/followup/kernel_rerun/extload.py check 10 2>&1 | tail -1)"
  CMD=($E/bin/vllm serve $M --served-model-name qwen --port 8000 --max-model-len 4096 --gpu-memory-utilization 0.85 --enable-lora --max-loras 16 --max-cpu-loras 64 --max-lora-rank 16 --lora-modules)
  for i in $(seq 0 31); do CMD+=("adapter_$(printf %04d $i)=$A/adapter_$(printf %04d $i)"); done
  setsid nohup env PATH=$E/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda "${CMD[@]}" > $O/server.log 2>&1 < /dev/null & SP=$!
  ok=0; for i in $(seq 1 900); do [ "$(curl -s -o /dev/null -w %{http_code} -m 3 localhost:8000/health)" = 200 ] && { ok=1; break; }; kill -0 $SP 2>/dev/null || break; sleep 3; done
  say "$tag ready=$ok"; [ $ok = 1 ] && { warm vllm http://localhost:8000 $tag; micro $tag vllm http://localhost:8000; }
  stop_server; say "done $tag";;
J) # dedicated-style rerun on runpod2: stock vs graph limit 256, 16 and 32 slots, 3 rounds, untraced
  declare -A CFG=( [S16]="16:" [C16]="16:--cuda-graph-max-bs-decode 256" [S32]="32:" [C32]="32:--cuda-graph-max-bs-decode 256" )
  rnd=0; for order in "S16 C16 S32 C32" "C32 S32 C16 S16" "C16 S32 S16 C32"; do rnd=$((rnd+1)); for c in $order; do
    tag=J_${c}-r$rnd; sl=${CFG[$c]%%:*}; EXTRA="${CFG[$c]#*:}"; PATCHPY=; XENV=
    start_sglang $tag main $sl 0 && { warm sglang http://localhost:30000 $tag; for rate in 20 40; do load $tag sglang http://localhost:30000 $rate 1; done; }
    stop_server; say "done $tag"; done; done;;
M0) # trigger step 2 pre-check: one traced start, 30 requests at 5 rps with rids; not analyzed for timing
  tag=M0_smoke; PATCHPY=; XENV=; EXTRA=
  start_sglang $tag main 16 1 && $PYL $R/scripts/loadgen.py --engine sglang --url http://localhost:30000 --tokenizer $M --rate 5 --requests 30 --warmup 0 --adapters 32 --popularity zipf --seed 1 --rid-prefix $tag --out $D/$tag/smoke.json | cut -c1-300 | tee -a $L
  stop_server; say "done $tag";;
M) # trigger step 2 (PLAN.md, option 13): stock main, 16 slots, traced with front-end events; absolute send times and rids
  for n in $(seq 1 ${NSTARTS:-10}); do tag=M_main-16-$n; PATCHPY=; XENV=; EXTRA=; RIDP=$tag
    start_sglang $tag main 16 1 && { warm sglang http://localhost:30000 $tag; for rate in 20 40; do load $tag sglang http://localhost:30000 $rate 1; done; say "$tag trace lines: $(cat $D/$tag/trace/*.jsonl | wc -l)"; }
    stop_server; say "done $tag"; done; RIDP=;;
N) # trigger step 3 (PLAN.md, option 14): part M with 120 busy-loop CPU processes during each start
  for n in $(seq 1 ${NSTARTS:-10}); do tag=N_main-16-cpu-$n; PATCHPY=; XENV=; EXTRA=; RIDP=$tag
    for i in $(seq 1 ${NBUSY:-120}); do python3 -c 'while True: pass' & echo $! >> /tmp/busy.pids; done; sleep 5; say "$tag load: $(cut -d' ' -f1-3 /proc/loadavg)"
    start_sglang $tag main 16 1 && { warm sglang http://localhost:30000 $tag; for rate in 20 40; do load $tag sglang http://localhost:30000 $rate 1; done; say "$tag load: $(cut -d' ' -f1-3 /proc/loadavg) trace lines: $(cat $D/$tag/trace/*.jsonl | wc -l)"; }
    stop_server; xargs kill < /tmp/busy.pids 2>/dev/null; rm -f /tmp/busy.pids; say "done $tag"; done; RIDP=;;
P) # trigger step 3 amendment: run this part under taskset -c 0-15; 5 pinned controls, then 10 starts with busy loops on cores 16-127
  for spec in pin-1 pin-2 pin-3 pin-4 pin-5 $(for n in $(seq 1 10); do echo pinload-$n; done); do tag=P_main-16-$spec; PATCHPY=; XENV=; EXTRA=; RIDP=$tag
    case $spec in pinload-*) for i in $(seq 1 120); do taskset -c 16-127 python3 -c 'while True: pass' & echo $! >> /tmp/busy.pids; done; sleep 5;; esac
    say "$tag affinity: $(taskset -pc $$ | cut -d: -f2) load: $(cut -d' ' -f1-3 /proc/loadavg)"
    start_sglang $tag main 16 1 && { warm sglang http://localhost:30000 $tag; for rate in 20 40; do load $tag sglang http://localhost:30000 $rate 1; done; say "$tag load: $(cut -d' ' -f1-3 /proc/loadavg) trace lines: $(cat $D/$tag/trace/*.jsonl | wc -l)"; }
    stop_server; [ -f /tmp/busy.pids ] && { xargs kill < /tmp/busy.pids 2>/dev/null; rm -f /tmp/busy.pids; }; say "done $tag"; done; RIDP=;;
Q) # trigger step 3 amendment 2: k busy loops (no pinning) in a 17.85-CPU quota container; cgroup throttle counters logged
  thr(){ say "$1 cpu.stat: $(tr '\n' ' ' < /sys/fs/cgroup/cpu/cpu.stat)"; }
  for rep in 1 2 3; do for k in 0 8 14; do tag=Q_main-16-k$k-$rep; PATCHPY=; XENV=; EXTRA=; RIDP=$tag
    for i in $(seq 1 $k); do python3 -c 'while True: pass' & echo $! >> /tmp/busy.pids; done; sleep 5; thr "$tag pre-start"
    start_sglang $tag main 16 1 && { warm sglang http://localhost:30000 $tag; for rate in 20 40; do thr "$tag pre-r$rate"; load $tag sglang http://localhost:30000 $rate 1; thr "$tag post-r$rate"; done; }
    stop_server; [ -f /tmp/busy.pids ] && { xargs kill < /tmp/busy.pids 2>/dev/null; rm -f /tmp/busy.pids; }; say "done $tag"; done; done; RIDP=;;
R0) # trigger step 4 pre-check: phase events appear; not analyzed for timing
  tag=R0_smoke; PATCHPY=; XENV="SGLANG_CLIFF_PHASES=1"; EXTRA=
  start_sglang $tag main 16 1 && $PYL $R/scripts/loadgen.py --engine sglang --url http://localhost:30000 --tokenizer $M --rate 5 --requests 30 --warmup 0 --adapters 32 --popularity zipf --seed 1 --rid-prefix $tag --out $D/$tag/smoke.json | cut -c1-200 | tee -a $L
  stop_server; XENV=; say "done $tag";;
R) # trigger step 4: part Q protocol with scheduler-loop phase timing, k in {0, 14}
  thr(){ say "$1 cpu.stat: $(tr '\n' ' ' < /sys/fs/cgroup/cpu/cpu.stat)"; }
  for rep in 1 2 3; do for k in 0 14; do tag=R_main-16-k$k-$rep; PATCHPY=; XENV="SGLANG_CLIFF_PHASES=1"; EXTRA=; RIDP=$tag
    for i in $(seq 1 $k); do python3 -c 'while True: pass' & echo $! >> /tmp/busy.pids; done; sleep 5; thr "$tag pre-start"
    start_sglang $tag main 16 1 && { warm sglang http://localhost:30000 $tag; for rate in 20 40; do thr "$tag pre-r$rate"; load $tag sglang http://localhost:30000 $rate 1; thr "$tag post-r$rate"; done; }
    stop_server; [ -f /tmp/busy.pids ] && { xargs kill < /tmp/busy.pids 2>/dev/null; rm -f /tmp/busy.pids; }; say "done $tag"; done; done; RIDP=; XENV=;;
S0) # trigger step 5 pre-check
  tag=S0_smoke; PATCHPY=; XENV="SGLANG_CLIFF_PHASES=1 SGLANG_CLIFF_SUBPHASES=1"; EXTRA=
  start_sglang $tag main 16 1 && $PYL $R/scripts/loadgen.py --engine sglang --url http://localhost:30000 --tokenizer $M --rate 5 --requests 30 --warmup 0 --adapters 32 --popularity zipf --seed 1 --rid-prefix $tag --out $D/$tag/smoke.json | cut -c1-200 | tee -a $L
  stop_server; XENV=; say "done $tag";;
S) # trigger step 5: sub-phase timing inside batch selection; k=0 once, k=14 three times
  thr(){ say "$1 cpu.stat: $(tr '\n' ' ' < /sys/fs/cgroup/cpu/cpu.stat)"; }
  for spec in k0-1 k14-1 k14-2 k14-3; do k=${spec%-*}; k=${k#k}; tag=S_main-16-$spec; PATCHPY=; XENV="SGLANG_CLIFF_PHASES=1 SGLANG_CLIFF_SUBPHASES=1"; EXTRA=; RIDP=$tag
    for i in $(seq 1 $k); do python3 -c 'while True: pass' & echo $! >> /tmp/busy.pids; done; sleep 5; thr "$tag pre-start"
    start_sglang $tag main 16 1 && { warm sglang http://localhost:30000 $tag; for rate in 20 40; do thr "$tag pre-r$rate"; load $tag sglang http://localhost:30000 $rate 1; thr "$tag post-r$rate"; done; }
    stop_server; [ -f /tmp/busy.pids ] && { xargs kill < /tmp/busy.pids 2>/dev/null; rm -f /tmp/busy.pids; }; say "done $tag"; done; RIDP=; XENV=;;
T) # step 6 experiment 1: 14 busy loops, stock against graph range 256, interleaved, 3 rounds
  thr(){ say "$1 cpu.stat: $(tr '\n' ' ' < /sys/fs/cgroup/cpu/cpu.stat)"; }
  for rep in 1 2 3; do for arm in stock cg256; do tag=T_main-16-k14-$arm-$rep; PATCHPY=; XENV="SGLANG_CLIFF_PHASES=1"; RIDP=$tag
    EXTRA=; [ $arm = cg256 ] && EXTRA="--cuda-graph-max-bs-decode 256"
    for i in $(seq 1 14); do python3 -c 'while True: pass' & echo $! >> /tmp/busy.pids; done; sleep 5; thr "$tag pre-start"
    start_sglang $tag main 16 1 && { warm sglang http://localhost:30000 $tag; for rate in 20 40; do thr "$tag pre-r$rate"; load $tag sglang http://localhost:30000 $rate 1; thr "$tag post-r$rate"; done; }
    stop_server; [ -f /tmp/busy.pids ] && { xargs kill < /tmp/busy.pids 2>/dev/null; rm -f /tmp/busy.pids; }; say "done $tag"; done; done; RIDP=; XENV=; EXTRA=;;
U) # step 6 experiment 2: decode step cost by batch size, closed loop at concurrency 32, 64, 96; no busy loops
  lc() { local O=$D/$1; echo "{\"conc\":$3,\"adapters\":$2,\"t_start\":$(date +%s.%N)," >> $O/marks.tmp
    $PYL $R/scripts/loadgen.py --engine sglang --url http://localhost:30000 --tokenizer $M --concurrency $3 --requests 400 --warmup 40 --adapters $2 --popularity zipf --seed 1 --out $O/c$3_a$2.json | cut -c1-300 | tee -a $L
    echo "\"t_end\":$(date +%s.%N)}" >> $O/marks.tmp; tr -d '\n' < $O/marks.tmp >> $O/marks.jsonl; echo >> $O/marks.jsonl; rm -f $O/marks.tmp; }
  for cfg in "U1_lora-stock:0::32 0" "U2_lora-cg256:0:--cuda-graph-max-bs-decode 256:32 0" "U3_nolora-stock:1::0" "U4_nolora-cg256:1:--cuda-graph-max-bs-decode 256:0"; do
    tag=${cfg%%:*}; r=${cfg#*:}; NOLORA=${r%%:*}; r=${r#*:}; EXTRA=${r%%:*}; ads=${r#*:}; PATCHPY=; XENV="SGLANG_CLIFF_PHASES=1 SGLANG_CLIFF_SUBPHASES=1"
    start_sglang $tag main 16 1 && { warm sglang http://localhost:30000 $tag; for a in $ads; do for c in 32 64 96; do lc $tag $a $c; done; done; }
    stop_server; say "done $tag"; done; NOLORA=0; XENV=; EXTRA=;;
V) # step 7: stock, cap48, cg256; 14 busy loops x2 rounds, then no busy loops x1
  thr(){ say "$1 cpu.stat: $(tr '\n' ' ' < /sys/fs/cgroup/cpu/cpu.stat)"; }
  for spec in k14-1 k14-2 k0-1; do k=${spec%-*}; k=${k#k}; for arm in stock cap48 cg256; do tag=V_main-16-$arm-$spec; PATCHPY=; XENV="SGLANG_CLIFF_PHASES=1"; RIDP=$tag
    case $arm in stock) EXTRA=;; cap48) EXTRA="--max-running-requests 48";; cg256) EXTRA="--cuda-graph-max-bs-decode 256";; esac
    for i in $(seq 1 $k); do python3 -c 'while True: pass' & echo $! >> /tmp/busy.pids; done; sleep 5; thr "$tag pre-start"
    start_sglang $tag main 16 1 && { warm sglang http://localhost:30000 $tag; for rate in 20 40; do thr "$tag pre-r$rate"; load $tag sglang http://localhost:30000 $rate 1; thr "$tag post-r$rate"; done; }
    stop_server; [ -f /tmp/busy.pids ] && { xargs kill < /tmp/busy.pids 2>/dev/null; rm -f /tmp/busy.pids; }; say "done $tag"; done; done; RIDP=; XENV=; EXTRA=;;
W0) # step 8 pre-check: patched server starts and logs the new default
  tag=W0_smoke; PATCHPY=/root/src/sglang-graphcap/python; XENV=; EXTRA=
  start_sglang $tag main 16 1 && $PYL $R/scripts/loadgen.py --engine sglang --url http://localhost:30000 --tokenizer $M --rate 5 --requests 30 --warmup 0 --adapters 32 --popularity zipf --seed 1 --out $D/$tag/smoke.json | cut -c1-200 | tee -a $L
  say "$tag: $(grep -o 'max_running_requests defaults to [0-9]* (was [0-9]*)' $D/$tag/server.log | head -1)"; stop_server; PATCHPY=; say "done $tag";;
W) # step 8: stock vs patch vs patch+cg256, k in {0,14}, 2 rounds
  thr(){ say "$1 cpu.stat: $(tr '\n' ' ' < /sys/fs/cgroup/cpu/cpu.stat)"; }
  for rep in 1 2; do for k in 0 14; do for arm in stock patch patch-cg256; do tag=W_main-16-$arm-k$k-$rep; XENV="SGLANG_CLIFF_PHASES=1"; RIDP=$tag
    case $arm in stock) PATCHPY=; EXTRA=;; patch) PATCHPY=/root/src/sglang-graphcap/python; EXTRA=;; patch-cg256) PATCHPY=/root/src/sglang-graphcap/python; EXTRA="--cuda-graph-max-bs-decode 256";; esac
    for i in $(seq 1 $k); do python3 -c 'while True: pass' & echo $! >> /tmp/busy.pids; done; sleep 5; thr "$tag pre-start"
    start_sglang $tag main 16 1 && { say "$tag: $(grep -o 'max_running_requests defaults to [0-9]* (was [0-9]*)' $D/$tag/server.log | head -1)"; warm sglang http://localhost:30000 $tag; for rate in 20 40; do thr "$tag pre-r$rate"; load $tag sglang http://localhost:30000 $rate 1; thr "$tag post-r$rate"; done; }
    stop_server; [ -f /tmp/busy.pids ] && { xargs kill < /tmp/busy.pids 2>/dev/null; rm -f /tmp/busy.pids; }; say "done $tag"; done; done; done; RIDP=; XENV=; EXTRA=; PATCHPY=;;
X0) # step 9 pre-check: greedy agreement and split smoke
  G=$R/followup/sglang_cliff/greedy_check.py
  for cfg in "stock::" "split:/root/src/sglang-split/python:" "cg256::--cuda-graph-max-bs-decode 256" "cg256b::--cuda-graph-max-bs-decode 256"; do
    arm=${cfg%%:*}; r=${cfg#*:}; PATCHPY=${r%%:*}; EXTRA=${r#*:}; tag=X0_$arm; XENV="SGLANG_CLIFF_PHASES=1"
    start_sglang $tag main 16 1 && { warm sglang http://localhost:30000 $tag; $PYL $G run http://localhost:30000 $D/$tag/greedy.json 96 | tee -a $L
      [ $arm = split ] && $PYL $R/scripts/loadgen.py --engine sglang --url http://localhost:30000 --tokenizer $M --concurrency 96 --requests 400 --warmup 40 --adapters 32 --popularity zipf --seed 1 --out $D/$tag/c96.json | cut -c1-300 | tee -a $L; }
    stop_server; say "done $tag"; done
  for p in "stock cg256" "split cg256" "cg256b cg256" "split stock"; do set -- $p; $PYL $G diff $D/X0_$1/greedy.json $D/X0_$2/greedy.json | tee -a $L; done; PATCHPY=; EXTRA=; XENV=;;
X) # step 9: stock, cap48, cg256, split, vllm; k in {0,14}; 2 rounds
  thr(){ say "$1 cpu.stat: $(tr '\n' ' ' < /sys/fs/cgroup/cpu/cpu.stat)"; }
  for rep in 1 2; do for k in 0 14; do for arm in stock cap48 cg256 split vllm; do tag=X_$arm-k$k-$rep; RIDP=; PATCHPY=; EXTRA=; XENV="SGLANG_CLIFF_PHASES=1"
    case $arm in cap48) EXTRA="--max-running-requests 48";; cg256) EXTRA="--cuda-graph-max-bs-decode 256";; split) PATCHPY=/root/src/sglang-split/python;; esac
    for i in $(seq 1 $k); do python3 -c 'while True: pass' & echo $! >> /tmp/busy.pids; done; sleep 5; thr "$tag pre-start"
    if [ $arm = vllm ]; then O=$D/$tag; mkdir -p $O; E=/root/envs/vllm
      CMD=($E/bin/vllm serve $M --served-model-name qwen --port 8000 --max-model-len 4096 --gpu-memory-utilization 0.85 --enable-lora --max-loras 16 --max-cpu-loras 64 --max-lora-rank 16 --lora-modules)
      for i in $(seq 0 31); do CMD+=("adapter_$(printf %04d $i)=$A/adapter_$(printf %04d $i)"); done
      say "start $tag host $(hostname) load $(cut -d' ' -f1-3 /proc/loadavg)"
      setsid nohup env PATH=$E/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda "${CMD[@]}" > $O/server.log 2>&1 < /dev/null & SP=$!
      ok=0; for i in $(seq 1 900); do [ "$(curl -s -o /dev/null -w %{http_code} -m 3 localhost:8000/health)" = 200 ] && { ok=1; break; }; kill -0 $SP 2>/dev/null || break; sleep 3; done; say "$tag ready=$ok"
      [ $ok = 1 ] && { warm vllm http://localhost:8000 $tag; for rate in 20 40; do thr "$tag pre-r$rate"; load $tag vllm http://localhost:8000 $rate 1; thr "$tag post-r$rate"; done; }
    else RIDP=$tag; start_sglang $tag main 16 1 && { warm sglang http://localhost:30000 $tag; for rate in 20 40; do thr "$tag pre-r$rate"; load $tag sglang http://localhost:30000 $rate 1; thr "$tag post-r$rate"; done; }; fi
    stop_server; [ -f /tmp/busy.pids ] && { xargs kill < /tmp/busy.pids 2>/dev/null; rm -f /tmp/busy.pids; }; say "done $tag"; done; done; done; RIDP=; XENV=; EXTRA=; PATCHPY=;;
Y0) # step 9 amendment: split v1 vs v2 step cost, c96 closed loop, sub-phase timing
  for spec in v1-1 v2-1 v1-2 v2-2; do v=${spec%-*}; tag=Y0_split$spec; EXTRA=; XENV="SGLANG_CLIFF_PHASES=1 SGLANG_CLIFF_SUBPHASES=1"
    PATCHPY=/root/src/sglang-split/python; [ $v = v2 ] && PATCHPY=/root/src/sglang-split2/python
    start_sglang $tag main 16 1 && { warm sglang http://localhost:30000 $tag; $PYL $R/scripts/loadgen.py --engine sglang --url http://localhost:30000 --tokenizer $M --concurrency 96 --requests 400 --warmup 40 --adapters 32 --popularity zipf --seed 1 --out $D/$tag/c96.json | cut -c1-300 | tee -a $L; }
    stop_server; say "done $tag"; done; PATCHPY=; XENV=;;
Y) # step 9 amendment: split v2 in part X protocol
  thr(){ say "$1 cpu.stat: $(tr '\n' ' ' < /sys/fs/cgroup/cpu/cpu.stat)"; }
  for rep in 1 2; do for k in 0 14; do tag=Y_splitv2-k$k-$rep; RIDP=$tag; PATCHPY=/root/src/sglang-split2/python; EXTRA=; XENV="SGLANG_CLIFF_PHASES=1"
    for i in $(seq 1 $k); do python3 -c 'while True: pass' & echo $! >> /tmp/busy.pids; done; sleep 5; thr "$tag pre-start"
    start_sglang $tag main 16 1 && { warm sglang http://localhost:30000 $tag; for rate in 20 40; do thr "$tag pre-r$rate"; load $tag sglang http://localhost:30000 $rate 1; thr "$tag post-r$rate"; done; }
    stop_server; [ -f /tmp/busy.pids ] && { xargs kill < /tmp/busy.pids 2>/dev/null; rm -f /tmp/busy.pids; }; say "done $tag"; done; done; RIDP=; XENV=; PATCHPY=;;
Z0) # PR prep smoke: rebased split on main, overlaid on the 0e9 env
  tag=Z0_splitmain; PATCHPY=/root/src/sglang-splitmain/python; EXTRA=; XENV="SGLANG_CLIFF_PHASES=1"
  start_sglang $tag main 16 1 && { warm sglang http://localhost:30000 $tag; $PYL $R/followup/sglang_cliff/greedy_check.py run http://localhost:30000 $D/$tag/greedy.json 96 | tee -a $L
    $PYL $R/scripts/loadgen.py --engine sglang --url http://localhost:30000 --tokenizer $M --concurrency 96 --requests 400 --warmup 40 --adapters 32 --popularity zipf --seed 1 --out $D/$tag/c96.json | cut -c1-300 | tee -a $L; }
  say "$tag errors in log: $(grep -ciE 'Traceback|Error' $D/$tag/server.log)"; stop_server; PATCHPY=; XENV=; say "done $tag";;
Z1) # PR prep accuracy: sgl-eval gsm8k greedy on stock, split (main), cg256
  for cfg in "stock::" "splitmain:/root/src/sglang-splitmain/python:" "cg256::--cuda-graph-max-bs-decode 256"; do
    arm=${cfg%%:*}; r=${cfg#*:}; PATCHPY=${r%%:*}; EXTRA=${r#*:}; tag=Z1_$arm; XENV="SGLANG_CLIFF_PHASES=1"
    start_sglang $tag main 16 1 && { /root/envs/sgleval/bin/sgl-eval run gsm8k --base-url http://127.0.0.1:30000/v1 --model qwen --temperature 0 --max-tokens 512 --seed 0 --num-threads 128 --out-dir $D/$tag/sgleval > $D/$tag/sgleval.log 2>&1; tail -5 $D/$tag/sgleval.log | tee -a $L; }
    stop_server; say "done $tag"; done; PATCHPY=; EXTRA=; XENV=;;
Z2) # debug: gsm8k on split v1, v2, v2+sync (all on 0e9)
  for cfg in "v1:/root/src/sglang-split/python" "v2:/root/src/sglang-split2/python" "v2sync:/root/src/sglang-split3/python"; do
    arm=${cfg%%:*}; PATCHPY=${cfg#*:}; EXTRA=; tag=Z2_$arm; XENV="SGLANG_CLIFF_PHASES=1"
    start_sglang $tag main 16 1 && { /root/envs/sgleval/bin/sgl-eval run gsm8k --base-url http://127.0.0.1:30000/v1 --model qwen --temperature 0 --max-tokens 512 --seed 0 --num-threads 128 --out-dir $D/$tag/sgleval > $D/$tag/sgleval.log 2>&1; tail -3 $D/$tag/sgleval.log | tee -a $L; }
    stop_server; say "done $tag"; done; PATCHPY=; XENV=;;
Z3) # debug: gsm8k + c96 timing on split v4
  tag=Z3_v4; PATCHPY=/root/src/sglang-split4/python; EXTRA=; XENV="SGLANG_CLIFF_PHASES=1"
  start_sglang $tag main 16 1 && { /root/envs/sgleval/bin/sgl-eval run gsm8k --base-url http://127.0.0.1:30000/v1 --model qwen --temperature 0 --max-tokens 512 --seed 0 --num-threads 128 --out-dir $D/$tag/sgleval > $D/$tag/sgleval.log 2>&1
    echo "{\"t_c96_start\":$(date +%s.%N)}" > $D/$tag/c96_mark.json
    $PYL $R/scripts/loadgen.py --engine sglang --url http://localhost:30000 --tokenizer $M --concurrency 96 --requests 400 --warmup 40 --adapters 32 --popularity zipf --seed 1 --out $D/$tag/c96.json | cut -c1-300 | tee -a $L; }
  stop_server; PATCHPY=; XENV=; say "done $tag";;
Z4) # debug isolation on split v4: coarse WAR barrier; overlap schedule off
  for cfg in "coarse:SGLANG_FORCE_COARSE_WAR_BARRIER=1:" "nooverlap::--disable-overlap-schedule"; do
    arm=${cfg%%:*}; r=${cfg#*:}; XX=${r%%:*}; EXTRA=${r#*:}; tag=Z4_v4$arm; PATCHPY=/root/src/sglang-split4/python; XENV="SGLANG_CLIFF_PHASES=1 $XX"
    start_sglang $tag main 16 1 && { /root/envs/sgleval/bin/sgl-eval run gsm8k --base-url http://127.0.0.1:30000/v1 --model qwen --temperature 0 --max-tokens 512 --seed 0 --num-threads 128 --out-dir $D/$tag/sgleval > $D/$tag/sgleval.log 2>&1; }
    stop_server; say "done $tag"; done; PATCHPY=; XENV=; EXTRA=;;
Z5) # debug: v5 (event cleared) and v2 + coarse barrier
  for cfg in "v5:/root/src/sglang-split5/python:" "v2coarse:/root/src/sglang-split2/python:SGLANG_FORCE_COARSE_WAR_BARRIER=1"; do
    arm=${cfg%%:*}; r=${cfg#*:}; PATCHPY=${r%%:*}; XX=${r#*:}; EXTRA=; tag=Z5_$arm; XENV="SGLANG_CLIFF_PHASES=1 $XX"
    start_sglang $tag main 16 1 && { /root/envs/sgleval/bin/sgl-eval run gsm8k --base-url http://127.0.0.1:30000/v1 --model qwen --temperature 0 --max-tokens 512 --seed 0 --num-threads 128 --out-dir $D/$tag/sgleval > $D/$tag/sgleval.log 2>&1
      echo "{\"t_c96_start\":$(date +%s.%N)}" > $D/$tag/c96_mark.json
      $PYL $R/scripts/loadgen.py --engine sglang --url http://localhost:30000 --tokenizer $M --concurrency 96 --requests 400 --warmup 40 --adapters 32 --popularity zipf --seed 1 --out $D/$tag/c96.json | cut -c1-300 | tee -a $L; }
    stop_server; say "done $tag"; done; PATCHPY=; XENV=;;
K) # Phase 3 option 8: raised graph limit plus overlap loading
  for n in 1 2 3; do tag=K_main-16-cg256-ovl-$n; PATCHPY=; XENV=; EXTRA="--cuda-graph-max-bs-decode 256 --enable-lora-overlap-loading"
    start_sglang $tag main 16 1 && { warm sglang http://localhost:30000 $tag; for rate in 20 40; do load $tag sglang http://localhost:30000 $rate 1; done; say "$tag trace lines: $(cat $D/$tag/trace/*.jsonl | wc -l)"; }
    stop_server; say "done $tag"; done;;
K2) # option 8 after amendment 3: raised graph limit, overlap loading, --max-loaded-loras 32
  for n in 1 2 3; do tag=K2_main-16-cg256-ovl-$n; PATCHPY=; XENV=; EXTRA="--cuda-graph-max-bs-decode 256 --enable-lora-overlap-loading --max-loaded-loras 32"
    start_sglang $tag main 16 1 && { warm sglang http://localhost:30000 $tag; for rate in 20 40; do load $tag sglang http://localhost:30000 $rate 1; done; say "$tag trace lines: $(cat $D/$tag/trace/*.jsonl | wc -l)"; }
    stop_server; say "done $tag"; done;;
L) # option 9.1: launch-only graph range check, no load
  for cfg in "L1_lora-default:lora:" "L2_nolora-085:nolora:--mem-fraction-static 0.85" "L3_nolora-default:nolora:"; do tag=${cfg%%:*}; rest=${cfg#*:}; kind=${rest%%:*}; mf=${rest#*:}; O=$D/$tag; mkdir -p $O
    busy && { say "REFUSED $tag: busy"; continue; }
    say "start $tag host $(hostname) load $(cut -d' ' -f1-3 /proc/loadavg)"
    CMD=(/root/envs/sglang_main/bin/python -m sglang.launch_server --model-path $M --served-model-name qwen --port 30000 --context-length 4096 $mf)
    if [ $kind = lora ]; then CMD+=(--enable-lora --max-lora-rank 16 --max-loras-per-batch 16 --lora-target-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj --lora-paths); for i in $(seq 0 31); do CMD+=("adapter_$(printf %04d $i)=$A/adapter_$(printf %04d $i)"); done; fi
    setsid nohup env PATH=/root/envs/sglang_main/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda SGLANG_CLIFF_NOTRACE=1 "${CMD[@]}" > $O/server.log 2>&1 < /dev/null & SP=$!
    ok=0; for i in $(seq 1 300); do [ "$(curl -s -o /dev/null -w %{http_code} -m 3 localhost:30000/health)" = 200 ] && { ok=1; break; }; kill -0 $SP 2>/dev/null || break; sleep 3; done
    say "$tag ready=$ok"; sleep 5; stop_server; say "done $tag"; done;;
esac
echo "DONE_$PART" >> $L
