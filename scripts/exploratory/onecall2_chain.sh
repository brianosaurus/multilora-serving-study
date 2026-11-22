#!/usr/bin/env bash
cd /root/study
while ! grep -q DONE logs/onecall_rep.summary 2>/dev/null; do sleep 15; done
export PATH=/usr/local/cuda/bin:/root/.cargo/bin:$PATH CUDA_HOME=/usr/local/cuda CUDA_COMPUTE_CAP=89
export CUDAFORGE_THREADS=2 CARGO_BUILD_JOBS=6 TMPDIR=/dev/shm/tmpbuild CARGO_TARGET_DIR=/root/study/target-mrs
S=logs/onecall2.summary; : > $S
echo "build start $(date -u +%T)" >> $S
(cd /root/study/build/mistral.rs && cargo build --release -p mistralrs-cli --features "cuda flash-attn" > /root/study/logs/cargo_onecall2.log 2>&1; echo "cargo-exit=$?" >> /root/study/$S)
if grep -q "cargo-exit=0" $S; then
  cp /root/study/target-mrs/release/mistralrs /root/study/bin/mistralrs-onecall2
  M=/dev/shm/models/Qwen2.5-1.5B-Instruct; A=/dev/shm/adapters; PYL=/root/study/venvs/vllm/bin/python; SC=/root/study/scripts
  OUT=/root/study/results/onecall2; mkdir -p $OUT
  for rep in 1 2 3; do for v in 0 1 2; do
    CMD=(/root/study/bin/mistralrs-onecall2 serve -m $M -p 1234 --kv-cache 8GiB --max-seqs 256 --lora-max-adapters 64)
    for i in $(seq 0 7); do CMD+=(--lora "adapter_$(printf %04d $i)=$A/adapter_$(printf %04d $i)"); done
    MRS_LORA_ONECALL=$v setsid nohup "${CMD[@]}" > logs/onecall2_$v.log 2>&1 < /dev/null &
    SP=$!
    for i in $(seq 1 100); do [ "$(curl -s -o /dev/null -w %{http_code} -m 3 localhost:1234/v1/models)" = 200 ] && break; sleep 3; done
    lg() { n=$1; shift; echo "rep$rep gate=$v $n: $($PYL $SC/loadgen.py --engine mistralrs --url http://localhost:1234 --tokenizer $M --requests 150 --warmup 20 --seed 5 --out $OUT/${n}_v${v}_r$rep.json "$@" 2>&1 | tail -1 | $PYL -c "import sys,json;d=json.loads(sys.stdin.read());print(d[\"ok\"],d[\"errors\"],\"tpot_p50\",round(d[\"tpot_ms\"][\"p50\"],1),\"ttft_p95\",round(d[\"ttft_ms\"][\"p95\"]),\"tok/s\",d[\"tok_per_s\"])" 2>&1 | tail -1)" >> $S; }
    lg warm --concurrency 8 --adapters 8 --popularity uniform
    lg a8_c8 --concurrency 8 --adapters 8 --popularity uniform
    lg a0_c32 --concurrency 32 --adapters 0
    lg a1_c32 --concurrency 32 --adapters 1
    lg a8_c32 --concurrency 32 --adapters 8 --popularity uniform
    kill $SP; sleep 6; pkill -9 -f "[m]istralrs-onecall2 serve"; sleep 4
  done; done
else echo "BUILD FAILED" >> $S; fi
echo DONE >> $S
