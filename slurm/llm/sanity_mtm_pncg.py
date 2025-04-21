import json
from forge.pncg import run_mtm_pncg

results = run_mtm_pncg(
    model_name='openai-community/gpt2',
    init_wandb=True,
    alpha=0.5,
    beta=1.0,
    p=1.0,
    seqlen=20,
    steps=50_000,
    seed=42,
    num_samples=512,
)

with open('dumps/llm/sanity_mtm_pncg.json', 'w') as f:
    json.dump(results, f)
