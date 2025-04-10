import json
import numpy as np
from tqdm import tqdm
from forge.ising import run_mtm_pncg

BETA, STEPS, P = 0.42, 1000, 1.0

mtm_results = []
for seqlen in [4, 8, 16]:
    print(f'Tuning seqlen={seqlen}')
    for num_samples in [4, 8, 16, 32]:
        for alpha in tqdm(np.geomspace(0.05, 64, num=10), desc=f'num_samples={num_samples}'):
            for seed in range(30):
                res = run_mtm_pncg(
                    alpha=alpha,
                    num_samples=num_samples,
                    seqlen=seqlen,
                    beta=BETA,
                    p=P,
                    steps=STEPS,
                    seed=seed,
                    quiet=True,
                )

                final_tvd = res['tvds'][-1]
                avg_tail_tvd = np.mean(res['tvds'][STEPS//2:])

                mtm_results.append({
                    'seqlen': seqlen,
                    'num_samples': num_samples,
                    'alpha': alpha,
                    'seed': seed,
                    'final_tvd': final_tvd,
                    'avg_tail_tvd': avg_tail_tvd,
                    'accept_rate': res['accept_rate'],
                    'wallclock': res['wallclock']
                })

with open('dumps/ising/tune_mtm_pncg.json', 'w') as f:
    json.dump(mtm_results, f)
