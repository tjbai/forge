import json
from forge.llm import run_pncg

results = run_pncg()

with open('dumps/llm/sanity.json', 'w') as f:
    json.dump(results, f)
