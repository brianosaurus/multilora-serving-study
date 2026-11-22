#!/usr/bin/env python3
"""Deterministic prompt sets for the differential study (no external data). usage: make_prompts.py OUTDIR
Writes prompts_main.jsonl (64 chat prompts), prompts_short.jsonl (16 trivially short raw prompts), prompts_stop.jsonl, and prints sha256s."""
import json, random, sys, hashlib, os
out = sys.argv[1]; os.makedirs(out, exist_ok=True); R = random.Random(20261122)
factual = ["What is the capital of Australia?", "Who wrote the novel Pride and Prejudice?", "What is the boiling point of water at sea level in degrees Celsius?",
 "Name the largest planet in the solar system.", "In what year did the first Moon landing take place?", "What is the chemical symbol for gold?",
 "Which ocean is the largest by area?", "Who painted the Mona Lisa?", "What is the square root of 144?", "What language is primarily spoken in Brazil?",
 "Which element has the atomic number 1?", "What is the longest river in Africa?", "Who developed the theory of general relativity?",
 "What is the freezing point of water in Fahrenheit?", "Name the author of the play Hamlet.", "What is the smallest prime number?"]
code = ["Write a Python function that returns the nth Fibonacci number iteratively.", "Write a Python function that checks whether a string is a palindrome.",
 "Write a SQL query that returns the ten most recent orders from a table named orders with a created_at column.", "Write a JavaScript function that debounces another function.",
 "Write a Python function that merges two sorted lists into one sorted list.", "Write a bash one-liner that counts the lines in every .txt file in a directory.",
 "Write a Python class implementing a stack with push, pop and peek.", "Write a C function that reverses a null-terminated string in place.",
 "Write a Python function that returns the most common word in a text.", "Write a regular expression that matches an ISO 8601 date such as 2026-11-22 and explain it briefly.",
 "Write a Python function that flattens a nested list of integers.", "Write a Go function that returns the sum of a slice of ints.",
 "Write a Python function to compute the greatest common divisor of two integers.", "Write a Rust function that returns the maximum of a slice of i32 values.",
 "Write a Python generator that yields the prime numbers below a given limit.", "Explain what a race condition is and give a short Python example."]
arith = []
for k in range(16):
    a, b, c = R.randint(12, 97), R.randint(3, 19), R.randint(2, 9)
    arith.append([f"A train travels {a*b} kilometres in {b} hours. At the same speed, how far does it travel in {b+c} hours? Show your steps.",
                  f"Maria buys {c} notebooks at ${a} each and pays with ${a*c+b}. How much change does she get? Explain.",
                  f"What is {a} times {b} plus {c}? Think step by step.", f"If x + {c} = {a}, what is x squared? Show the work."][k % 4])
passages = [
 "The village of Harlow sits at the mouth of a narrow river. For three centuries its economy depended on salt, which was dried in shallow pans along the shore and carried inland by mule. When the railway arrived in 1871, the salt trade collapsed within a decade, because cheaper mined salt could now reach the cities by train. The pans were abandoned, and the villagers turned to fishing and, later, to guiding visitors through the marshes. Today the old salt houses are a small museum, and a festival in September commemorates the last mule driver, who died in 1936 at the age of ninety-one.",
 "A lighthouse keeper named Ines Calder kept a daily log for forty-two years. Most entries record the weather and the oil used, but a few mention ships that passed too close to the rocks. In February of 1958 she wrote that a freighter had lost power and drifted for six hours before a tug reached it. She had lit every lamp she owned and burned the spare oil to keep the beam steady. The crew later sent her a brass compass, which she kept on the windowsill beside the logbook until she retired in 1974.",
 "Photosynthesis converts light energy into chemical energy stored in sugars. In the light-dependent reactions, chlorophyll in the thylakoid membranes absorbs photons and splits water, releasing oxygen and producing ATP and NADPH. In the Calvin cycle, which takes place in the stroma, the enzyme rubisco fixes carbon dioxide into a three-carbon compound, and the ATP and NADPH from the first stage are used to build glucose. Plants regulate how much carbon dioxide enters by opening and closing pores called stomata, which also lose water vapour to the air.",
 "The committee met on a Tuesday to decide how to spend a budget surplus of forty thousand dollars. Three proposals were on the table: repaving the parking lot, buying new laptops for the library, or funding a summer reading programme for children. After two hours of debate the members agreed to split the money, with fifteen thousand going to the laptops and twenty-five thousand to the reading programme, and to postpone the parking lot until the next fiscal year when more funds would be available."]
questions = ["Summarize the passage in two sentences.", "What is the main event described, and what was its consequence?", "List the key facts in the passage as three bullet points.", "Write one question that a reader could answer from the passage, then answer it."]
main = []
for q in factual: main.append({"cat": "factual", "messages": [{"role": "user", "content": q}]})
for q in arith: main.append({"cat": "reasoning", "messages": [{"role": "user", "content": q}]})
for q in code: main.append({"cat": "code", "messages": [{"role": "user", "content": q}]})
for i, p in enumerate(passages):
    for j, q in enumerate(questions): main.append({"cat": "passage", "messages": [{"role": "user", "content": p + "\n\n" + q}]})
assert len(main) == 64, len(main)
for i, m in enumerate(main): m["id"] = i
short = [{"id": i, "cat": "short", "text": t} for i, t in enumerate(["Hi", "The", "1+1=", "Paris", "Yes", "Hello,", "def f(", "Once", "A B C", "42", "It is", "OK", "<", "Q:", "The cat", "x ="])]
stop = [{"id": 0, "stop": ["\n"]}, {"id": 1, "stop": ["."]}, {"id": 2, "stop": ["the"]}, {"id": 3, "stop": ["\n\n", "Answer", "###", "END"]}, {"id": 4, "stop": ["zzzqqq"]}, {"id": 5, "stop": [" a"]}]
def w(name, rows):
    p = os.path.join(out, name)
    with open(p, "w") as f:
        for r in rows: f.write(json.dumps(r, sort_keys=True) + "\n")
    print(name, len(rows), hashlib.sha256(open(p, "rb").read()).hexdigest())
w("prompts_main.jsonl", main); w("prompts_short.jsonl", short); w("prompts_stop.jsonl", stop)
