#!/usr/bin/env python3
"""Reference continuations and teacher-forced log-probability matrices for the routing follow-up.

Candidates: base and adapters 0 to 3. For every prompt and candidate: the fp32 greedy continuation of 128 tokens.
Then, for each model variant v (fp32 and bf16) and each prompt p: the log-probability of the 128 forced tokens of
every candidate's continuation under variant v. Output: JSON with tokens and the matrices.
"""
import json, sys, time
import torch, transformers, peft
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS, OUT = sys.argv[1], sys.argv[2]
THREADS = int(sys.argv[3]) if len(sys.argv) > 3 else 48
torch.set_num_threads(THREADS)
DEV = "cuda"
import os
M, A = "/root/k/model", os.environ.get("ADAPTER_DIR", "/root/k/adapters")
CANDS = ["base", "adapter_0000", "adapter_0001", "adapter_0002", "adapter_0003"]
N = 128
prompts = json.load(open(PROMPTS))
tok = AutoTokenizer.from_pretrained(M)
pids = [tok(p).input_ids for p in prompts]
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.0f}s]", *a, flush=True)

def load(dtype):
    base = AutoModelForCausalLM.from_pretrained(M, dtype=dtype).to(DEV).eval()
    m = PeftModel.from_pretrained(base, f"{A}/{CANDS[1]}", adapter_name=CANDS[1])
    for n in CANDS[2:]:
        m.load_adapter(f"{A}/{n}", adapter_name=n)
    return m

def under(m, c, fn):
    if c == "base":
        with m.disable_adapter():
            return fn()
    m.set_adapter(c)
    return fn()

def logprobs(m, v, seqs):
    ids = torch.tensor(seqs, device=DEV)
    with torch.inference_mode():
        logits = under(m, v, lambda: m(input_ids=ids).logits)
    out = []
    for b in range(ids.shape[0]):
        lp = torch.log_softmax(logits[b, :-1].float(), dim=-1)
        g = lp.gather(-1, ids[b, 1:].unsqueeze(-1)).squeeze(-1)
        out.append(g[-N:].cpu().tolist())
    return out

result = {"candidates": CANDS, "n_tokens": N, "prompts": prompts, "prompt_ids": pids,
          "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
          "threads": THREADS, "adapter_dir": A, "device": DEV, "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32)}
m32 = load(torch.float32)
log("fp32 model loaded")
tokens = []
for p, ids in enumerate(pids):
    row = []
    for c in CANDS:
        def gen():
            return m32.generate(torch.tensor([ids], device=DEV), max_new_tokens=N, min_new_tokens=N, do_sample=False,
                                pad_token_id=tok.eos_token_id)
        o = under(m32, c, gen)
        row.append(o[0, len(ids):].cpu().tolist())
    tokens.append(row)
    log("generated prompt", p)
result["tokens"] = tokens   # tokens[p][c]
seqs = lambda p: [pids[p] + tokens[p][c] for c in range(len(CANDS))]
def matrix(m, name):
    res = {}
    for v in CANDS:
        res[v] = [logprobs(m, v, seqs(p)) for p in range(len(pids))]   # [p][c][t]
        log(name, "variant", v, "done")
    return res
result["L32"] = matrix(m32, "fp32")
json.dump(result, open(OUT, "w"))
del m32
torch.cuda.empty_cache()
mbf = load(torch.bfloat16)
log("bf16 model loaded")
result["Lbf"] = matrix(mbf, "bf16")
json.dump(result, open(OUT, "w"))
log("wrote", OUT)
