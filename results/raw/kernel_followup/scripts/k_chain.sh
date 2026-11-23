#!/usr/bin/env bash
set -u
cd /root/k; L=/root/k/chain.log; : > $L
say() { echo "$(date -u +%T) $*" >> $L; }
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda CUDA_COMPUTE_CAP=89
export CARGO_BUILD_JOBS=6 CUDAFORGE_THREADS=2 TMPDIR=/root/k/tmp CARGO_TARGET_DIR=/root/k/target
cd /root/b3/mistral.rs
git apply --check /root/k/grouped.patch >> $L 2>&1 || { say "PATCH DOES NOT APPLY"; echo CHAIN_DONE >> $L; exit 1; }
git apply /root/k/grouped.patch; say "patch applied: $(git diff --stat | tail -1)"
say "test start"; cargo test --release -p mistralrs-quant --features cuda --lib lora::dynamic >> /root/k/cargo_test.log 2>&1; say "test exit $?"
grep -E "^test |test result|error(\[|:)" /root/k/cargo_test.log | tail -25 >> $L
if ! grep -q "test result: ok" /root/k/cargo_test.log; then say "TESTS DID NOT PASS"; echo CHAIN_DONE >> $L; exit 1; fi
say "build start"; cargo build --release -p mistralrs-cli --features cuda >> /root/k/cargo_grouped.log 2>&1; say "build exit $?"
cp /root/k/target/release/mistralrs /root/k/mrs-grouped && say "grouped binary $(ls -la /root/k/mrs-grouped | awk '{print $5}')"
M=/root/k/model; A=/root/k/adapters; PYL=/root/k/venv/bin/python; SC=/root/k/scripts; OUT=/root/k/ab; mkdir -p $OUT; S=/root/k/ab.summary; : > $S
for rep in 1 2 3; do for bin in base grouped; do
  CMD=(/root/k/mrs-$bin serve -m $M -p 1234 --kv-cache 8GiB --max-seqs 256 --lora-max-adapters 64)
  for i in $(seq 0 7); do CMD+=(--lora "adapter_$(printf %04d $i)=$A/adapter_$(printf %04d $i)"); done
  setsid nohup "${CMD[@]}" > /root/k/server_${bin}_$rep.log 2>&1 < /dev/null &
  SP=$!; for i in $(seq 1 100); do [ "$(curl -s -o /dev/null -w %{http_code} -m 3 localhost:1234/v1/models)" = 200 ] && break; sleep 3; done
  lg() { n=$1; shift; echo "rep$rep $bin $n: $($PYL $SC/loadgen.py --engine mistralrs --url http://localhost:1234 --tokenizer $M --requests 150 --warmup 20 --seed 5 --out $OUT/${n}_${bin}_r$rep.json "$@" 2>&1 | tail -1 | $PYL -c "import sys,json;d=json.loads(sys.stdin.read());print(d['ok'],d['errors'],'tpot_p50',round(d['tpot_ms']['p50'],1),'ttft_p95',round(d['ttft_ms']['p95']),'tok/s',d['tok_per_s'])" 2>&1 | tail -1)" >> $S; }
  lg warm --concurrency 8 --adapters 8 --popularity uniform
  lg a8_c8 --concurrency 8 --adapters 8 --popularity uniform
  lg a0_c32 --concurrency 32 --adapters 0
  lg a1_c32 --concurrency 32 --adapters 1
  lg a2_c32 --concurrency 32 --adapters 2 --popularity uniform
  lg a4_c32 --concurrency 32 --adapters 4 --popularity uniform
  lg a8_c32 --concurrency 32 --adapters 8 --popularity uniform
  kill $SP; sleep 6; kill -9 $SP 2>/dev/null; pkill -9 -f "mrs-$bin serve"; sleep 4
done; done
echo AB_DONE >> $S; say "CHAIN DONE"; echo CHAIN_DONE >> $L
