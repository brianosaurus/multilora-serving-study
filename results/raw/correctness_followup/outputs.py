import asyncio, aiohttp, json, sys
PROMPTS = ["The capital of France is", "Write a haiku about the sea.", "Explain what a mutex is in one sentence.",
 "List three prime numbers greater than 50.", "Translate 'good morning' into Spanish.", "What is 17 times 23?",
 "Name a famous physicist and one discovery.", "Complete: To be or not to be,", "Describe a rainbow briefly.",
 "Give a synonym for 'quick'.", "What does HTTP stand for?", "Write a short greeting for a new neighbour.",
 "Why is the sky blue?", "Name two programming languages.", "Spell the word 'necessary' backwards.", "Summarize the plot of Cinderella in one line."]
async def one(s, prompt, adapter):
    body = {"model": "default", "prompt": prompt, "max_tokens": 24, "temperature": 0, "adapter": adapter}
    async with s.post("http://localhost:1234/v1/completions", json=body) as r:
        d = await r.json()
    return d["choices"][0]["text"] if "choices" in d else "ERR:" + json.dumps(d)[:120]
async def main():
    out = {}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as s:
        for p in PROMPTS:
            names = [f"adapter_{i:04d}" for i in range(8)]
            res = await asyncio.gather(*[one(s, p, n) for n in names])
            for n, t in zip(names, res): out[f"{p}|{n}"] = t
    json.dump(out, open(sys.argv[1], "w"))
asyncio.run(main())
