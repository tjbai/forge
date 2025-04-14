import json
from forge.llm import run_mtm_pncg

results = run_mtm_pncg(
    alpha=4.0,
    beta=1.0,
    p=1.0,
    bsz=1,
    seqlen=20,
    steps=50_000,
    seed=42,
    num_samples=8,
)

with open('dumps/llm/sanity_mtm_pncg.json', 'w') as f:
    json.dump(results, f)
