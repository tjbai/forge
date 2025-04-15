import json
from forge.pncg import run_pncg

beta = 1.0

results = run_pncg(
    model_name='openai-community/gpt2',
    init_wandb=True,
    alpha=4.0,
    beta=beta,
    p=1.0,
    bsz=1,
    seqlen=20,
    steps=50_000,
    seed=42,
)

with open(f'dumps/llm/sanity_pncg_beta-{beta}.json', 'w') as f:
    json.dump(results, f)
