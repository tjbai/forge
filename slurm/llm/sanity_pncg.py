import json
from forge.pncg import run_pncg

results = run_pncg(
    alpha=4.0,
    beta=1.0,
    p=1.0,
    bsz=1,
    seqlen=20,
    steps=50_000,
    seed=42,
)

with open('dumps/llm/sanity_pncg.json', 'w') as f:
    json.dump(results, f)
