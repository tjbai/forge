import json
from tqdm import tqdm
from forge.pncg import run_mtm_pncg

results = []
for i in tqdm(range(50)):
    results.append(run_mtm_pncg(
        model_name='openai-community/gpt2',
        init_wandb=False,
        num_samples=512,
        alpha=0.2,
        beta=1.0,
        seqlen=20,
        steps=5000,
        seed=i,
    ))
    with open('dumps/llm/best_mtm_pncg.json', 'w') as f:
        json.dump(results, f)
