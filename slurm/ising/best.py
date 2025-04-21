import json
from tqdm import tqdm
from forge.ising import run_pncg, run_mtm_pncg, run_iw_mtm_pncg
from collections import defaultdict

BETA, STEPS, P = 0.42, 1000, 1.0

all = defaultdict(list)
for seqlen, alpha in zip([4, 8, 16], [64, 1.2, 0.5]):
    for seed in tqdm(range(30)):
        res = run_pncg(
            alpha=alpha,
            seqlen=seqlen,
            beta=BETA,
            p=P,
            steps=STEPS,
            seed=seed,
            quiet=True,
        )
        all[seqlen].append(res)

with open('dumps/ising/best_pncg.json', 'w') as f:
    json.dump(dict(all), f)

all = defaultdict(list)
for seqlen, alpha, num_samples in zip(
    [4, 8, 16],
    [28.9, 64, 28.9],
    [16, 32, 32],
):
    for seed in tqdm(range(30)):
        res = run_mtm_pncg(
            alpha=alpha,
            seqlen=seqlen,
            num_samples=num_samples,
            beta=BETA,
            p=P,
            steps=STEPS,
            seed=seed,
            quiet=True,
        )
        all[seqlen].append(res)

with open('dumps/ising/best_mtm_pncg.json', 'w') as f:
    json.dump(dict(all), f)

all = defaultdict(list)
for seqlen, alpha, num_samples in zip(
    [4, 8, 16],
    [28.9, 64, 28.9],
    [16, 32, 32],
):
    for seed in tqdm(range(30)):
        res = run_iw_mtm_pncg(
            alpha=alpha,
            seqlen=seqlen,
            num_samples=num_samples,
            beta=BETA,
            p=P,
            steps=STEPS,
            seed=seed,
            quiet=True,
        )
        all[seqlen].append(res)

with open('dumps/ising/best_iw_mtm_pncg.json', 'w') as f:
    json.dump(dict(all), f)
