#!/usr/bin/env python3
"""Generate synthetic PEFT LoRA adapters: rank r, all linear layers, nonzero B.

Each adapter gets its own seed. A is drawn from N(0, 1/fan_in) and B from N(0, b_std^2),
both from a seeded generator, so the same seed always gives the same adapter. PEFT's
default initializes B to zero, which would make every adapter a no-op, so B is set
explicitly. b_std has no default: it is chosen by calibrating against the PEFT
reference only, before any engine is tested.
"""
import argparse
import hashlib
import json
import os

import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, help="path to the base model directory")
    ap.add_argument("--out", required=True, help="output directory for the adapters")
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--b-std", type=float, required=True)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16)
    cfg = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        target_modules=TARGETS,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, cfg)
    save_dtype = getattr(torch, args.dtype)
    base_weights = os.path.join(args.base, "model.safetensors")
    manifest = {
        "base": os.path.abspath(args.base),
        "base_weights_sha256": sha256(base_weights) if os.path.exists(base_weights) else None,
        "rank": args.rank,
        "alpha": args.alpha,
        "b_std": args.b_std,
        "dtype": args.dtype,
        "targets": TARGETS,
        "init": "A ~ N(0, 1/fan_in), B ~ N(0, b_std^2), seeded per adapter",
        "versions": {
            "torch": torch.__version__,
            "peft": __import__("peft").__version__,
            "transformers": __import__("transformers").__version__,
        },
        "adapters": [],
    }

    for i in range(args.count):
        seed = args.first_seed + i
        gen = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for name, p in peft_model.named_parameters():
                if "lora_A" in name:
                    p.copy_(torch.randn(p.shape, generator=gen) / p.shape[1] ** 0.5)
                elif "lora_B" in name:
                    p.copy_(torch.randn(p.shape, generator=gen) * args.b_std)
        d = os.path.join(args.out, f"adapter_{seed:04d}")
        peft_model.save_pretrained(d)
        wpath = os.path.join(d, "adapter_model.safetensors")
        tensors = {k: v.to(save_dtype).contiguous() for k, v in load_file(wpath).items()}
        save_file(tensors, wpath, metadata={"format": "pt"})
        manifest["adapters"].append(
            {
                "name": f"adapter_{seed:04d}",
                "seed": seed,
                "path": os.path.abspath(d),
                "bytes": os.path.getsize(wpath),
                "sha256": sha256(wpath),
            }
        )
        print(f"wrote {d} ({manifest['adapters'][-1]['bytes'] / 1e6:.1f} MB)")

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("manifest:", os.path.join(args.out, "manifest.json"))


if __name__ == "__main__":
    main()
