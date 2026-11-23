import sys, json, torch, numpy as np
sys.path.insert(0, "/root/k/scripts")
import loadgen
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
M = "/root/k/model"; A = "/root/k/adapters"
tok = AutoTokenizer.from_pretrained(M)
rng = np.random.default_rng(5)
prompts = loadgen.make_prompts(tok, 170, 5, 256, 0)
picks = [f"adapter_{i:04d}" for i in rng.choice(8, size=170, p=np.full(8, 1 / 8))]
base = AutoModelForCausalLM.from_pretrained(M, torch_dtype=torch.float32)
model = None
EOS, IM_END = 151643, 151645
rows = {}
targets = [2, 136, 20, 32, 106, 5, 50]
for name in sorted({picks[t] for t in targets}):
    model = PeftModel.from_pretrained(model.base_model.model if model is not None and hasattr(model, "base_model") and False else base, f"{A}/{name}", adapter_name=name) if model is None else model
    if name not in model.peft_config:
        model.load_adapter(f"{A}/{name}", adapter_name=name)
    model.set_adapter(name)
    model.eval()
    for t in [t for t in targets if picks[t] == name]:
        ids = tok(prompts[t], return_tensors="pt").input_ids
        with torch.no_grad():
            lp = torch.log_softmax(model(ids).logits[0, -1].float(), dim=-1)
        top = torch.topk(lp, 5)
        rows[t] = {"adapter": name, "top5": [(int(i), round(float(v), 3)) for v, i in zip(top.values, top.indices)],
                   "eos_logprob": round(float(lp[EOS]), 3), "im_end_logprob": round(float(lp[IM_END]), 3),
                   "eos_rank": int((lp > lp[EOS]).sum()) + 1}
json.dump(rows, open("/root/k/ref_firsttoken.json", "w"), indent=1)
for t in targets: print(t, rows[t])
