"""Exit non-zero unless the base weights and all 32 adapters match the study's manifest."""

import hashlib
import json
import os
import sys


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


adapter_dir, manifest_path, base_dir = sys.argv[1:4]
manifest = json.load(open(manifest_path))
base = sha256(os.path.join(base_dir, "model.safetensors"))
ok = base == manifest["base_weights_sha256"]
print("base weights", "match" if ok else f"MISMATCH {base}")
matched = 0
# The manifest covers more adapters than these runs use; check seeds 0 to 31.
expected = [e for e in manifest["adapters"] if e["seed"] < 32]
for entry in expected:
    got = sha256(os.path.join(adapter_dir, entry["name"], "adapter_model.safetensors"))
    if got == entry["sha256"]:
        matched += 1
    else:
        print("MISMATCH", entry["name"], got)
print(f"adapters matching the manifest: {matched} of {len(expected)}")
sys.exit(0 if ok and len(expected) == 32 and matched == 32 else 1)
