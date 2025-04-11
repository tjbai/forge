import json
import numpy as np
from tqdm import tqdm
from forge.ising import run_pncg

BETA, STEPS, P = 0.42, 1000, 1.0

mtm_results = []
for seqlen in [4, 8, 16]:
    print(f'Tuning seqlen={seqlen}')
    for alpha in tqdm(np.geomspace(0.05, 64, num=10)):
        for seed in range(30):
            res = run_pncg(
                alpha=alpha,
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
                'alpha': alpha,
                'seed': seed,
                'final_tvd': float(final_tvd),
                'avg_tail_tvd': float(avg_tail_tvd),
                'accept_rate': float(res['accept_rate']),
                'wallclock': res['wallclock']
            })

        with open('dumps/ising/tune_pncg.json', 'w') as f:
            json.dump(mtm_results, f)
