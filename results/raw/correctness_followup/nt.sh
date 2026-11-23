#!/usr/bin/env bash
cd /root/k; M=/root/k/model; A=/root/k/adapters; OUT=/root/k/nt; mkdir -p $OUT; : > /root/k/nt.log
for tag in base grouped base2 grouped2; do
  b=${tag%2}; [ $tag = base2 ] && b=base2; [ $tag = grouped2 ] && b=grouped
  CMD=(/root/k/mrs-$b serve -m $M -p 1234 --kv-cache 8GiB --max-seqs 256 --lora-max-adapters 64)
  for i in $(seq 0 7); do CMD+=(--lora "adapter_$(printf %04d $i)=$A/adapter_$(printf %04d $i)"); done
  setsid nohup "${CMD[@]}" > /root/k/server5_$tag.log 2>&1 < /dev/null &
  SP=$!; for i in $(seq 1 100); do [ "$(curl -s -o /dev/null -w %{http_code} -m 3 localhost:1234/v1/models)" = 200 ] && break; sleep 3; done
  /root/k/venv/bin/python /root/k/nt.py $OUT/$tag.json >> /root/k/nt.log 2>&1; echo "$tag done $(date -u +%T)" >> /root/k/nt.log
  kill $SP; sleep 6; kill -9 $SP 2>/dev/null; pkill -9 -f "mrs-$b serve"; sleep 4
done
echo NT_DONE >> /root/k/nt.log
