#!/usr/bin/env python3
"""Noise floor of the adapter identity criterion.

Scores a mathematically equivalent but numerically different implementation (the adapter
merged into the base weights, bf16) with the same criterion the engines are held to. It
shows how often a correct implementation fails the criterion because of numerics alone.
Needs the GPU idle. Diagnostic only: it does not change any criterion.
"""
import argparse
import json
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from check_engine import MAX_TOKENS, PASS_THRESHOLD, score


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapters", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ref = json.load(open(args.reference))
    prompts = ref["prompts"]
    tok = AutoTokenizer.from_pretrained(args.base)
    ref_text = {v: [tok.decode(ids, skip_special_tokens=True) for ids in outs]
                for v, outs in ref["variants"].items()}
    adapters = [v for v in ref["variants"] if v != "base"]

    results = []
    for a in adapters:
        model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16).cuda().eval()
        model = PeftModel.from_pretrained(model, os.path.join(args.adapters, a)).merge_and_unload()
        for i, p in enumerate(prompts):
            ids = tok(p, return_tensors="pt").input_ids.cuda()
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=MAX_TOKENS, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
            r = score(text, a, i, ref_text)
            r.update({"adapter": a, "prompt": i, "text": text})
            results.append(r)
        del model
        torch.cuda.empty_cache()
        print("done", a)

    correct = sum(r["correct"] for r in results)
    per = {a: sum(r["correct"] for r in results if r["adapter"] == a) / len(prompts) for a in adapters}
    print(f"merged-weights bf16 vs unmerged PEFT reference: {correct}/{len(results)} = "
          f"{correct / len(results):.3f} (threshold {PASS_THRESHOLD}), per-adapter {per}")
    json.dump({"correct": correct, "n": len(results), "accuracy": correct / len(results),
               "per_adapter": per, "details": results}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
