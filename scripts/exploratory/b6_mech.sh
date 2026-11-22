#!/usr/bin/env bash
# B6 mechanism probe: mistral.rs, 8 adapters loaded, closed loop at different concurrency and adapter mixes
M=/dev/shm/models/Qwen2.5-1.5B-Instruct; A=/dev/shm/adapters; PYL=/root/study/venvs/vllm/bin/python; SC=/root/study/scripts
OUT=/root/study/results/b6_mech; mkdir -p $OUT; export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
CMD=(/root/study/bin/mistralrs serve -m $M -p 1234 --kv-cache 8GiB --max-seqs 256 --lora-max-adapters 64)
for i in $(seq 0 7); do CMD+=(--lora "adapter_$(printf %04d $i)=$A/adapter_$(printf %04d $i)"); done
nohup setsid "${CMD[@]}" > /root/study/logs/b6_mech.log 2>&1 < /dev/null &
pid=$!
for i in $(seq 1 100); do [ "$(curl -s -o /dev/null -w %{http_code} -m 3 localhost:1234/v1/models)" = 200 ] && break; sleep 3; done
lg() { name=$1; shift; echo "$name: $($PYL $SC/loadgen.py --engine mistralrs --url http://localhost:1234 --tokenizer $M --requests ${NREQ:-200} --warmup 20 --seed 5 --out $OUT/$name.json "$@" 2>&1 | tail -1 | cut -c1-330)" >> /root/study/logs/b6_mech.log.summary; }
echo start $(date -u +%T) > /root/study/logs/b6_mech.log.summary
lg warm --concurrency 4 --adapters 0
lg base_c8 --concurrency 8 --adapters 0
lg one_c8 --concurrency 8 --adapters 1
lg ad8_c1 --concurrency 1 --adapters 8 --popularity uniform
lg one_c1 --concurrency 1 --adapters 1
lg ad8_c2 --concurrency 2 --adapters 8 --popularity uniform
lg ad8_c4 --concurrency 4 --adapters 8 --popularity uniform
nvidia-smi dmon -s puc -d 1 -c 40 > $OUT/ad8_c8_dmon.txt 2>&1 &
( sleep 8; top -H -b -n 2 -d 3 -w 200 | head -60 > $OUT/ad8_c8_top.txt ) &
lg ad8_c8 --concurrency 8 --adapters 8 --popularity uniform
wait
echo DONE $(date -u +%T) >> /root/study/logs/b6_mech.log.summary
kill $pid; sleep 6; pkill -9 -f "[m]istralrs serve"
