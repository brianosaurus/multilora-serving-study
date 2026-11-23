#!/usr/bin/env python3
"""Part A, A3: alignment, token-id identity, runtime dtypes, TF32 and attention implementation, logit magnitudes.
Uses the same loading code as refs.py and the token ids saved in refs_out.json."""
import json, sys, os
import numpy as np, torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
M, A = "/root/k/model", "/root/k/adapters"; DEV = "cuda"
R = json.load(open(sys.argv[1])); C = R["candidates"]; N = R["n_tokens"]
tok = AutoTokenizer.from_pretrained(M)
def load(dtype):
    base = AutoModelForCausalLM.from_pretrained(M, dtype=dtype).to(DEV).eval()
    m = PeftModel.from_pretrained(base, f"{A}/{C[1]}", adapter_name=C[1])
    for n in C[2:]: m.load_adapter(f"{A}/{n}", adapter_name=n)
    return m
def under(m, c, fn):
    if c == "base":
        with m.disable_adapter(): return fn()
    m.set_adapter(c); return fn()
models = {"fp32": load(torch.float32), "bf16": load(torch.bfloat16)}
print("== runtime settings")
print("torch", torch.__version__, "| matmul.allow_tf32:", torch.backends.cuda.matmul.allow_tf32, "| cudnn.allow_tf32:", torch.backends.cudnn.allow_tf32,
      "| float32_matmul_precision:", torch.get_float32_matmul_precision())
for k, m in models.items():
    bm = m.base_model.model
    emb = bm.model.embed_tokens.weight
    lin = bm.model.layers[0].self_attn.q_proj          # LoRA-wrapped Linear
    la, lb = lin.lora_A[C[1]].weight, lin.lora_B[C[1]].weight
    print(f"{k}: base embed dtype {emb.dtype} | q_proj base weight dtype {lin.base_layer.weight.dtype} | lora_A dtype {la.dtype} | lora_B dtype {lb.dtype} "
          f"| lora scaling {lin.scaling[C[1]]} | attn implementation {bm.config._attn_implementation} | lm_head dtype {bm.lm_head.weight.dtype}")
def forward(m, v, ids):
    with torch.inference_mode():
        return under(m, v, lambda: m(input_ids=ids).logits)
def tf_logprobs(logits, ids):
    lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)      # same operation as refs.py
    return lp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
print("\n== alignment, ids and dtypes on 2 requests (adapter_0002 with prompts 0 and 9)")
for p in (0, 9):
    v = "adapter_0002"; c = C.index(v)
    pid = R["prompt_ids"][p]; T = R["tokens"][p][c]
    ids_arr = pid + T
    ids = torch.tensor([ids_arr], device=DEV)
    ids_ref = torch.tensor([R["prompt_ids"][p] + R["tokens"][p][c]])        # what refs.py built, rebuilt from the saved JSON
    print(f"prompt {p}: token-id arrays identical between runs (prompt+continuation, {len(ids_arr)} ids): {bool(torch.equal(ids.cpu(), ids_ref))}")
    out = {}
    for k, m in models.items():
        lg = forward(m, v, ids)
        out[k] = (lg, tf_logprobs(lg, ids)[0, -N:])
        print(f"   {k}: logits dtype {lg.dtype}; after .float() {lg.float().dtype}; log_softmax output dtype {torch.log_softmax(lg[:, :-1].float(), -1).dtype}")
    print("   pos  token    fp32_teacher  bf16_teacher  fp32_prefix-only  bf16_prefix-only   (prefix-only = forward on prompt+T[:t], read the last position)")
    for t in range(10):
        pre = torch.tensor([pid + T[:t]], device=DEV)
        vals = []
        for k, m in models.items():
            last = torch.log_softmax(forward(m, v, pre)[0, -1].float(), -1)[T[t]].item()
            vals.append(last)
        print(f"   {t:3d}  {T[t]:6d}  {out['fp32'][1][t].item():12.4f}  {out['bf16'][1][t].item():12.4f}  {vals[0]:16.4f}  {vals[1]:16.4f}")
print("\n== logit magnitudes (all 128 forced positions of each request)")
for v in ("base", "adapter_0000", "adapter_0002"):
    c = C.index(v); p = 0
    ids = torch.tensor([R["prompt_ids"][p] + R["tokens"][p][c]], device=DEV)
    for k, m in models.items():
        lg = forward(m, v, ids)[0, -N - 1:-1].float()
        mx = lg.abs().max(dim=-1).values
        print(f"{v:13s} {k}: max|logit| per position median {mx.median().item():8.1f}, p95 {mx.quantile(0.95).item():8.1f}, max {mx.max().item():8.1f}; "
              f"std of logits at a position, median {lg.std(dim=-1).median().item():6.2f}")
print("bf16 spacing between adjacent representable values: [64,128): 0.5, [128,256): 1.0, [256,512): 2.0")
print("\n== the 10 largest per-token differences: ranks under each model")
top = json.load(open(sys.argv[2]))
for e in top:
    v = e["adapter"]; c = C.index(v); p = e["prompt"]; t = e["pos"]
    ids = torch.tensor([R["prompt_ids"][p] + R["tokens"][p][c]], device=DEV)
    pos = len(R["prompt_ids"][p]) + t - 1                 # logits row that predicts token t
    row = {}
    for k, m in models.items():
        lg = forward(m, v, ids)[0, pos].float()
        row[k] = (int((lg > lg[e["token"]]).sum().item()) + 1, float(lg[e["token"]]), float(lg.max()), int(lg.argmax()))
    print(f"{v} p{p:2d} t{t:3d} token {e['token']:6d} {tok.decode([e['token']])!r:14s} | fp32: rank {row['fp32'][0]:5d} logit {row['fp32'][1]:7.1f} (max {row['fp32'][2]:6.1f}, argmax {row['fp32'][3]}) "
          f"| bf16: rank {row['bf16'][0]:5d} logit {row['bf16'][1]:7.1f} (max {row['bf16'][2]:6.1f}, argmax {row['bf16'][3]})")
