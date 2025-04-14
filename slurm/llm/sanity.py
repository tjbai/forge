import json
from forge.llm import run_pncg

results = run_pncg(steps=50_000, seqlen=20)

with open('dumps/llm/sanity.json', 'w') as f:
    json.dump(results, f)
