#!/usr/bin/env bash
# A2: mistral.rs v0.9.5 (cuda flash-attn build, commit 465191d) engine runs at scale 0.02 (plan: followup/routing/PLAN.md)
cd /root/k/routing; L=mistral_a2.log; : > $L; PY=/root/e/sglang/bin/python
say() { echo "$(date -u +%T) $*" >> $L; }
state() { echo "load $(cut -d' ' -f1-3 /proc/loadavg) | gpu $(nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw,utilization.gpu,memory.used --format=csv,noheader | tr -d ' ')"; }
say "host $(hostname); GPU procs: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l); binary $(sha256sum /root/rerun/mrs-v095-fa | cut -c1-16); $(state)"
[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { say "REFUSED: GPU not idle"; echo MISTRAL_A2_DONE >> $L; exit 1; }
A=/root/k/routing/adapters_s0.02
CMD=(/root/rerun/mrs-v095-fa serve -m /root/k/model -p 1234 --kv-cache 8GiB --max-seqs 256 --lora-max-adapters 64)
for i in 0 1 2 3; do CMD+=(--lora "adapter_000$i=$A/adapter_000$i"); done
"${CMD[@]}" > mistral_server_a2.log 2>&1 &
SP=$!
for i in $(seq 1 120); do [ "$(curl -s -o /dev/null -w %{http_code} -m 3 localhost:1234/v1/models)" = 200 ] && break; kill -0 $SP 2>/dev/null || break; sleep 3; done
say "mistral.rs up? $(curl -s -o /dev/null -w %{http_code} -m 3 localhost:1234/v1/models); $(state)"
$PY engine_probe.py --engine mistralrs --url http://localhost:1234 --refs refs_s0.02.json --tokenizer /root/k/model --mode serial --limit 3 --out mistral_a2_smoke.json >> $L 2>&1
$PY check_smoke.py refs_s0.02.json mistral_a2_smoke.json >> $L 2>&1
GATE=$($PY - <<PY
import json
d=json.load(open("mistral_a2_smoke.json")); s=d["stats"]
bad=sum(1 for c in d["calls"] if c["logp"] is None or any(x is None for x in c["logp"]))
frac=(s["missing_topn"]+s["dropped_roundtrip"])/max(1,s["positions"])
print("PASS" if s["errors"]==0 and bad==0 and frac<=0.05 else "FAIL", f"errors={s['errors']} incomplete_calls={bad} missing_or_dropped_fraction={frac:.4f} positions={s['positions']}")
PY
)
say "smoke gate: $GATE"
case "$GATE" in PASS*) ;; *) say "smoke failed: stopping before the full runs"; kill $SP; sleep 5; echo MISTRAL_A2_DONE >> $L; exit 1;; esac
for run in "mixed mistral_a2_mixed" "serial mistral_a2_serial" "serial mistral_a2_serial_repeat"; do
  set -- $run; say "start $2: $(state)"
  LIM=""; [ "$2" = mistral_a2_serial_repeat ] && LIM="--limit 20"
  $PY engine_probe.py --engine mistralrs --url http://localhost:1234 --refs refs_s0.02.json --tokenizer /root/k/model --mode $1 $LIM --out $2.json >> $L 2>&1
  say "end $2: $(state)"
done
kill $SP; sleep 8; kill -9 $SP 2>/dev/null
say "server stopped; gpu mem $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
for f in mistral_a2_serial mistral_a2_mixed mistral_a2_serial_repeat; do $PY analyze_engine.py refs_s0.02.json $f.json >> mistral_a2_analysis.txt 2>&1; echo >> mistral_a2_analysis.txt; done
$PY - >> mistral_a2_analysis.txt <<PY
import json, numpy as np
a=json.load(open("mistral_a2_serial.json")); b=json.load(open("mistral_a2_serial_repeat.json"))
da={(c["p"],c["a"],c["c"]):c["logp"] for c in a["calls"]}; db={(c["p"],c["a"],c["c"]):c["logp"] for c in b["calls"]}
m=[float(np.mean(np.abs(np.array(da[k])-np.array(db[k])))) for k in db if k in da and da[k] and db[k] and None not in da[k] and None not in db[k]]
print(f"self-consistency (context, not part of the rule): mistral.rs serial pass vs its repeat over {len(m)} calls: mean |diff| median {np.median(m):.4f}, p95 {np.percentile(m,95):.4f}, max {max(m):.4f}; stand-in noise 0.0201")
for name,d in (("serial",a),("mixed",json.load(open("mistral_a2_mixed.json"))),("serial repeat",b)):
    s=d["stats"]; print(f"{name}: positions {s['positions']}, missing from the top-20 list {s['missing_topn']} ({s['missing_topn']/max(1,s['positions']):.2%}), dropped by the text round trip {s['dropped_roundtrip']} ({s['dropped_roundtrip']/max(1,s['positions']):.2%}), errors {s['errors']}")
PY
say "MISTRAL_A2_DONE"; echo MISTRAL_A2_DONE >> $L
