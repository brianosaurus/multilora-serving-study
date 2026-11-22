#!/usr/bin/env python3
"""First-token log-probability identity check (the amended criterion in docs/PLAN.md).

Subcommands:
  reference   compute the first-token top-20 log-probs for the base model and each adapter
              (PEFT, unmerged, bf16) and for each adapter merged into the weights (the
              noise-floor implementation). Needs the GPU idle.
  noisefloor  score the merged implementation with the same metric (offline).
  engine      query a running engine and score it.

Distance to a variant = mean absolute difference in log-probability over the variant's
reference top-10 tokens; a token the other side does not return counts as that side's
lowest returned log-probability. A request passes if the nearest variant is the adapter
it asked for.
"""
import argparse
import json
import os
import random
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

TOPK_REF = 20
TOPK_USE = 10
PASS_THRESHOLD = 0.90


def decode_tokens(tok, ids):
    return [tok.decode([i]) for i in ids]


def distance(engine_top: dict, ref_tokens: list, ref_lps: list, k: int = TOPK_USE) -> float:
    floor = min(engine_top.values())
    total = 0.0
    for s, rlp in list(zip(ref_tokens, ref_lps))[:k]:
        total += abs(engine_top.get(s, floor) - rlp)
    return total / k


def nearest(engine_top: dict, variants_ref: dict, prompt_idx: int):
    dists = {}
    for v, per_prompt in variants_ref.items():
        entry = per_prompt[prompt_idx]
        dists[v] = distance(engine_top, entry["tokens"], entry["lps"])
    best = min(dists, key=dists.get)
    return best, dists


def cmd_reference(args):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompts = json.load(open(args.prompts))
    names = [a["name"] for a in json.load(open(os.path.join(args.adapters, "manifest.json")))["adapters"]]
    tok = AutoTokenizer.from_pretrained(args.base)

    def topk(model, prompt):
        ids = tok(prompt, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            logits = model(ids).logits[0, -1].float()
        lp = torch.log_softmax(logits, dim=-1)
        vals, idx = lp.topk(TOPK_REF)
        return {"tokens": decode_tokens(tok, idx.tolist()), "lps": vals.tolist()}

    def load_base():
        return AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16).cuda().eval()

    model = load_base()
    model = PeftModel.from_pretrained(model, os.path.join(args.adapters, names[0]), adapter_name=names[0])
    for n in names[1:]:
        model.load_adapter(os.path.join(args.adapters, n), adapter_name=n)
    variants = {}
    with model.disable_adapter():
        variants["base"] = [topk(model, p) for p in prompts]
    for n in names:
        model.set_adapter(n)
        variants[n] = [topk(model, p) for p in prompts]
        print("reference done", n)
    del model
    torch.cuda.empty_cache()

    merged = {}
    for n in names:
        m = PeftModel.from_pretrained(load_base(), os.path.join(args.adapters, n)).merge_and_unload()
        merged[n] = [topk(m, p) for p in prompts]
        del m
        torch.cuda.empty_cache()
        print("merged done", n)

    json.dump({"prompts": prompts, "variants": variants, "merged": merged,
               "topk_ref": TOPK_REF, "topk_use": TOPK_USE},
              open(args.out, "w"))
    print("wrote", args.out)


def summarize(rows, label):
    n = len(rows)
    ok = sum(r["correct"] for r in rows)
    per = {}
    for r in rows:
        per.setdefault(r["adapter"], []).append(r["correct"])
    mean_d = sum(r["dist_correct"] for r in rows) / n
    out = {"label": label, "requests": n, "correct": ok, "accuracy": ok / n,
           "per_adapter": {a: sum(v) / len(v) for a, v in per.items()},
           "mean_distance_to_correct": mean_d, "pass": ok / n >= PASS_THRESHOLD}
    print(f"{label:<28} {ok}/{n} = {ok / n:.3f}  pass={out['pass']}  "
          f"mean dist to correct={mean_d:.3f}  per-adapter={ {a: round(v, 2) for a, v in out['per_adapter'].items()} }")
    return out


def cmd_noisefloor(args):
    ref = json.load(open(args.reference))
    adapters = [v for v in ref["variants"] if v != "base"]
    rows = []
    for a in adapters:
        for i in range(len(ref["prompts"])):
            top = dict(zip(ref["merged"][a][i]["tokens"], ref["merged"][a][i]["lps"]))
            best, dists = nearest(top, ref["variants"], i)
            rows.append({"adapter": a, "prompt": i, "nearest": best, "correct": best == a,
                         "dist_correct": dists[a], "dists": dists})
    out = summarize(rows, "noise floor (merged bf16)")
    json.dump({"summary": out, "rows": rows}, open(args.out, "w"), indent=1)
    print("noise floor valid (>= 0.90):", out["pass"])


def cmd_fp32(args):
    """Supplementary, non-gating: unmerged PEFT in fp32 scored against the bf16 reference."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ref = json.load(open(args.reference))
    prompts = ref["prompts"]
    adapters = [v for v in ref["variants"] if v != "base"]
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float32).cuda().eval()
    model = PeftModel.from_pretrained(model, os.path.join(args.adapters, adapters[0]), adapter_name=adapters[0])
    for n in adapters[1:]:
        model.load_adapter(os.path.join(args.adapters, n), adapter_name=n)
    rows = []
    for a in adapters:
        model.set_adapter(a)
        for i, p in enumerate(prompts):
            ids = tok(p, return_tensors="pt").input_ids.cuda()
            with torch.no_grad():
                logits = model(ids).logits[0, -1].float()
            vals, idx = torch.log_softmax(logits, dim=-1).topk(TOPK_REF)
            top = dict(zip(decode_tokens(tok, idx.tolist()), vals.tolist()))
            best, dists = nearest(top, ref["variants"], i)
            rows.append({"adapter": a, "prompt": i, "nearest": best, "correct": best == a,
                         "dist_correct": dists[a], "dists": dists})
    out = summarize(rows, "fp32 unmerged (supplementary)")
    json.dump({"summary": out, "rows": rows, "gating": False}, open(args.out, "w"), indent=1)


def build_request(engine, adapter, prompt, base_model):
    body = {"prompt": prompt, "max_tokens": 1, "temperature": 0, "logprobs": TOPK_REF}
    if engine == "vllm":
        body["model"] = adapter or base_model
    elif engine == "sglang":
        body["model"] = base_model
        if adapter:
            body["lora_path"] = adapter
    elif engine == "mistralrs":
        body["model"] = "default"
        if adapter:
            body["adapter"] = adapter
    return body


def post_top(url, body, timeout=180):
    req = urllib.request.Request(url + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    lp = d["choices"][0].get("logprobs")
    if not lp:
        return None
    if "content" in lp:
        # mistral.rs layout: top entries carry the token id in "token" and the text in "bytes"
        entries = lp["content"][0].get("top_logprobs") if lp["content"] else None
        if not entries:
            return None
        out = {}
        for e in entries:
            b = e["bytes"]
            out[bytes(b).decode("utf-8", "replace") if isinstance(b, list) else b] = e["logprob"]
        return out
    if not lp.get("top_logprobs") or not lp["top_logprobs"][0]:
        return None
    return lp["top_logprobs"][0]


def cmd_engine(args):
    ref = json.load(open(args.reference))
    prompts = ref["prompts"]
    adapters = [v for v in ref["variants"] if v != "base"]
    tasks = [(a, i) for a in adapters for i in range(len(prompts))]
    mixed_tasks = list(tasks)
    random.Random(0).shuffle(mixed_tasks)

    probe = post_top(args.url, build_request(args.engine, None, prompts[0], args.base_model_name))
    if probe is None:
        res = {"engine": args.engine, "checkable": False,
               "note": "the engine returned no top log-probabilities for the first token"}
        json.dump(res, open(args.out, "w"), indent=1)
        print(f"{args.engine}: NOT CHECKABLE (no top log-probabilities returned)")
        return

    def call(t):
        a, i = t
        return post_top(args.url, build_request(args.engine, a, prompts[i], args.base_model_name))

    def run(label, ts, parallel):
        t0 = time.time()
        if parallel:
            with ThreadPoolExecutor(max_workers=len(ts)) as ex:
                tops = list(ex.map(call, ts))
        else:
            tops = [call(t) for t in ts]
        rows = []
        for (a, i), top in zip(ts, tops):
            if top is None:
                rows.append({"adapter": a, "prompt": i, "nearest": None, "correct": False,
                             "dist_correct": float("nan"), "dists": {}})
                continue
            best, dists = nearest(top, ref["variants"], i)
            rows.append({"adapter": a, "prompt": i, "nearest": best, "correct": best == a,
                         "dist_correct": dists[a], "dists": dists})
        s = summarize([r for r in rows if r["dist_correct"] == r["dist_correct"]] or rows,
                      f"{args.engine} {label}")
        s["seconds"] = round(time.time() - t0, 1)
        return s, rows

    s_sum, s_rows = run("serial", tasks, False)
    m_sum, m_rows = run("mixed", mixed_tasks, True)
    res = {"engine": args.engine, "checkable": True, "threshold": PASS_THRESHOLD,
           "serial": s_sum, "mixed": m_sum, "overall_pass": s_sum["pass"] and m_sum["pass"],
           "rows": {"serial": s_rows, "mixed": m_rows}}
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"{args.engine} OVERALL first-token pass={res['overall_pass']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reference")
    r.add_argument("--base", required=True)
    r.add_argument("--adapters", required=True)
    r.add_argument("--prompts", required=True)
    r.add_argument("--out", required=True)
    n = sub.add_parser("noisefloor")
    n.add_argument("--reference", required=True)
    n.add_argument("--out", required=True)
    f = sub.add_parser("fp32")
    f.add_argument("--base", required=True)
    f.add_argument("--adapters", required=True)
    f.add_argument("--reference", required=True)
    f.add_argument("--out", required=True)
    e = sub.add_parser("engine")
    e.add_argument("--engine", required=True, choices=["vllm", "sglang", "mistralrs"])
    e.add_argument("--url", required=True)
    e.add_argument("--reference", required=True)
    e.add_argument("--base-model-name", default="qwen")
    e.add_argument("--out", required=True)
    args = ap.parse_args()
    {"reference": cmd_reference, "noisefloor": cmd_noisefloor, "fp32": cmd_fp32, "engine": cmd_engine}[args.cmd](args)


if __name__ == "__main__":
    main()
