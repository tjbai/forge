import json
from tqdm import tqdm
from forge.pncg import run_pncg

results = []
for i in tqdm(range(50)):
    results.append(run_pncg(
        model_name='openai-community/gpt2',
        init_wandb=False,
        alpha=3.5,
        beta=1.0,
        p=1.0,
        bsz=1,
        seqlen=20,
        steps=5000,
        seed=i,
    ))
    if (i + 1) % 10 == 0:
        with open('dumps/llm/best_pncg.json', 'w') as f:
            json.dump(results, f)
