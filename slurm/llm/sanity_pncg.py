import json
from forge.pncg import run_pncg

results = run_pncg(
    model_name='openai-community/gpt2',
    init_wandb=True,
    run_name='pncg_p-1',
    alpha=4.0,
    beta=1.0,
    p=1.0,
    bsz=1,
    seqlen=20,
    steps=50_000,
    seed=42,
)
