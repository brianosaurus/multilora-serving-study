#!/usr/bin/env python3
"""Item C cross-endpoint comparison from results/C_repro/repro.json (computed, not by hand).
Passes: i = temperature 0 only; ii = plus the model's defaults; rep = repeat of i. Order of requests per endpoint: i, ii, rep."""
import json, pathlib
d = json.load(open(pathlib.Path(__file__).parent / "results" / "C_repro" / "repro.json"))
T = {ep: d[ep]["texts"] for ep in d}
ne = lambda a, b: [k for k, (x, y) in enumerate(zip(a, b)) if x != y]
rows = [
 ("chat (ii) vs completions (ii)", ne(T["chat"]["ii"], T["completions"]["ii"])),
 ("chat repeat vs completions (ii)", ne(T["chat"]["repeat"], T["completions"]["ii"])),
 ("completions (i) vs generate (i)", ne(T["completions"]["i"], T["generate"]["i"])),
 ("completions (ii) vs generate (ii)", ne(T["completions"]["ii"], T["generate"]["ii"])),
 ("chat (i) vs chat (ii)", ne(T["chat"]["i"], T["chat"]["ii"])),
 ("chat (i) vs chat repeat", ne(T["chat"]["i"], T["chat"]["repeat"])),
 ("chat (ii) vs chat repeat", ne(T["chat"]["ii"], T["chat"]["repeat"])),
]
print("| Comparison (of 64 prompts) | prompts that differ |\n|---|---|")
for n, x in rows: print(f"| {n} | {len(x)} |")
a, b = set(rows[4][1]), set(rows[5][1])
print(f"\nchat (i) differs from (ii) on {len(a)} prompts and from the repeat on {len(b)}; the two sets are {'the same prompts' if a == b else 'different'}: {sorted(a)}")
print("chat 'ii' equals chat 'repeat' on", 64 - len(rows[6][1]), "of 64 prompts")
print("chat (ii) equals completions (ii) on", 64 - len(rows[0][1]), "of 64")
