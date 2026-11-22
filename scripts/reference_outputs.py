#!/usr/bin/env python3
"""PEFT reference: greedy outputs for the base model and for each adapter.

Uses Hugging Face transformers with PEFT (unmerged adapters, bf16, one prompt at a
time, no padding). Needs the GPU idle. The output is the ground truth that the
engines are checked against.
"""
import argparse
import json
import os

import peft
import torch
import transformers
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapters", required=True, help="directory holding adapter_* and manifest.json")
    ap.add_argument("--prompts", required=True, help="JSON list of prompt strings")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--new-tokens", type=int, default=32)
    ap.add_argument("--changed-within", type=int, default=8,
                    help="window used for the 'adapter changed the output' rate")
    args = ap.parse_args()

    with open(args.prompts) as f:
        prompts = json.load(f)
    with open(os.path.join(args.adapters, "manifest.json")) as f:
        manifest = json.load(f)
    names = [a["name"] for a in manifest["adapters"]]

    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16).cuda().eval()
    first = names[0]
    model = PeftModel.from_pretrained(model, os.path.join(args.adapters, first), adapter_name=first)
    for n in names[1:]:
        model.load_adapter(os.path.join(args.adapters, n), adapter_name=n)

    def generate(prompt: str) -> list[int]:
        ids = tok(prompt, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=args.new_tokens, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        return out[0, ids.shape[1]:].tolist()

    result = {
        "prompts": prompts,
        "new_tokens": args.new_tokens,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                     "peft": peft.__version__},
        "manifest": manifest,
        "variants": {},
    }
    base_out = []
    with model.disable_adapter():
        for p in prompts:
            base_out.append(generate(p))
    result["variants"]["base"] = base_out
    for n in names:
        model.set_adapter(n)
        result["variants"][n] = [generate(p) for p in prompts]
        print("done", n)

    w = args.changed_within
    for n in names:
        changed = sum(1 for a, b in zip(result["variants"][n], base_out) if a[:w] != b[:w])
        print(f"{n}: output differs from base within first {w} tokens on {changed}/{len(prompts)} prompts")
        result.setdefault("changed_vs_base", {})[n] = {"within": w, "changed": changed, "of": len(prompts)}

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
