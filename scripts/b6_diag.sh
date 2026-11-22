#!/usr/bin/env bash
# B6 diagnostic: mistral.rs default settings under rate-10 load, with GPU utilization sampled alongside.
set -u
M=/dev/shm/models/Qwen2.5-1.5B-Instruct; PYL=/root/study/venvs/vllm/bin/python; SC=/root/study/scripts
OUT=/root/study/results/b6; mkdir -p $OUT /root/study/logs
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
run_variant() { # name extra-args...
  name=$1; shift
  echo "=== B6 variant $name start $(date -u +%T) args: $*"
  nohup setsid /root/study/bin/mistralrs serve -m $M -p 1234 --kv-cache 8GiB "$@" > /root/study/logs/b6_$name.log 2>&1 < /dev/null &
  pid=$!
  for i in $(seq 1 100); do [ "$(curl -s -o /dev/null -w "%{http_code}" -m 3 http://localhost:1234/v1/models)" = 200 ] && break; sleep 3; done
  $PYL $SC/loadgen.py --engine mistralrs --url http://localhost:1234 --tokenizer $M --rate 10 --requests 150 --warmup 0 --adapters 0 --seed 998 --out $OUT/${name}_warm.json > /dev/null 2>&1
  nvidia-smi dmon -s puc -d 1 -c 50 > $OUT/${name}_dmon.txt 2>&1 &
  dm=$!
  $PYL $SC/loadgen.py --engine mistralrs --url http://localhost:1234 --tokenizer $M --rate 10 --requests 300 --warmup 50 --adapters 0 --seed 1 --out $OUT/${name}_r10.json 2>&1 | tail -1 | cut -c1-300
  wait $dm 2>/dev/null
  pkill -f "[r]elease/mistralrs|[b]in/mistralrs serve"; kill $pid 2>/dev/null; sleep 6; pkill -9 -f "[m]istralrs serve" 2>/dev/null; sleep 3
}
run_variant default
run_variant decode1 --max-decode-steps-before-prefill 1
run_variant decode64 --max-decode-steps-before-prefill 64
echo B6_DIAG_DONE
