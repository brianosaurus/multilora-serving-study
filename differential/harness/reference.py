#!/usr/bin/env python3
"""Reference generation + teacher-forced logprobs with Hugging Face transformers (bf16, sdpa, batch size 1, greedy).
usage: reference.py MODEL_DIR NAME OUTPREFIX PROMPTS_DIR   -> OUTPREFIX.main.jsonl, OUTPREFIX.short.jsonl, OUTPREFIX.info.json"""
import json, sys, time, platform
import torch, transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
mdir, name, outp, pdir = sys.argv[1:5]
VARIANT = sys.argv[5] if len(sys.argv) > 5 else "raw"   # raw: repetition_penalty=1.0 explicit (true greedy). hfdefault: HF applies the model generation_config (e.g. Qwen repetition_penalty 1.1)
tok = AutoTokenizer.from_pretrained(mdir); model = AutoModelForCausalLM.from_pretrained(mdir, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
eos = set([model.generation_config.eos_token_id] if isinstance(model.generation_config.eos_token_id, int) else (model.generation_config.eos_token_id or []))
if tok.eos_token_id is not None: eos.add(tok.eos_token_id)
@torch.no_grad()
def run(prompt_ids, max_new):
    ids = torch.tensor([prompt_ids]).cuda()
    kw = dict(repetition_penalty=1.0) if VARIANT == "raw" else {}
    out = model.generate(ids, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None, top_k=None, pad_token_id=tok.eos_token_id or 0, **kw)
    gen = out[0, len(prompt_ids):].tolist()
    full = torch.tensor([prompt_ids + gen]).cuda()
    logits = model(full).logits[0].float()            # [T, V]
    lsm = torch.log_softmax(logits, dim=-1)
    L = len(prompt_ids); lp, top = [], []
    for i, g in enumerate(gen):
        row = lsm[L - 1 + i]; lp.append(row[g].item())
        v, ix = torch.topk(row, 20); top.append([[int(a), float(b)] for a, b in zip(ix.tolist(), v.tolist())])
    return gen, lp, top, ("eos" if gen and gen[-1] in eos else "length")
t0 = time.time()
with open(outp + ".main.jsonl", "w") as f:
    for line in open(f"{pdir}/prompts_main.jsonl"):
        r = json.loads(line)
        ids = tok.apply_chat_template(r["messages"], add_generation_prompt=True, tokenize=True)
        ids = ids["input_ids"] if isinstance(ids, dict) or hasattr(ids, "keys") else ids
        gen, lp, top, fin = run(list(ids), 128)
        f.write(json.dumps({"id": r["id"], "cat": r["cat"], "prompt_ids": list(ids), "gen_ids": gen, "lp": lp, "top20": top, "finish": fin}) + "\n"); f.flush()
with open(outp + ".short.jsonl", "w") as f:
    for line in open(f"{pdir}/prompts_short.jsonl"):
        r = json.loads(line); ids = tok(r["text"], add_special_tokens=True)["input_ids"]
        gen, lp, top, fin = run(list(ids), 32)
        f.write(json.dumps({"id": r["id"], "prompt_ids": list(ids), "gen_ids": gen, "lp": lp, "top20": top, "finish": fin}) + "\n")
json.dump({"model": name, "model_dir": mdir, "torch": torch.__version__, "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(0),
           "dtype": "bfloat16", "attn": "sdpa", "variant": VARIANT, "generation_config": model.generation_config.to_dict(), "secs": round(time.time() - t0, 1), "python": platform.python_version(), "chat_template_head": (tok.chat_template or "")[:200]}, open(outp + ".info.json", "w"), indent=1)
print("DONE", name, round(time.time() - t0, 1), "s")
