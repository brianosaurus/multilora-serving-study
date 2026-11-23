import json
old = [json.load(open("b3out/old.json")), json.load(open("b3out/old_again.json"))] + [json.load(open(f"b3out/old_r{r}.json")) for r in range(1, 5)]
new = [json.load(open("b3out/new.json"))] + [json.load(open(f"b3out/new_r{r}.json")) for r in range(1, 5)]
varying = [k for k in old[0] if len({r[k] for r in old}) > 1]
inside = [k for k in varying if all(n[k] in {r[k] for r in old} for n in new)]
newvals = {k: {n[k] for n in new} for k in varying}
print(f"completions that vary across the 6 old launches: {len(varying)}; for how many is the new build's output (same in all 5 launches: {all(len(v)==1 for v in newvals.values())}) one of the values the old build produced: {len(inside)}/{len(varying)}")
for k in varying[:3]: print("  ", repr(k[:42]), "old produced", len({r[k] for r in old}), "distinct outputs; new produced", len(newvals[k]))
stable_same = sum(1 for k in old[0] if k not in varying and old[0][k] == new[0][k])
print(f"completions stable across all old launches: {128-len(varying)}; of those, identical to new: {stable_same}/{128-len(varying)}")
