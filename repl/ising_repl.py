# %%
import json
import pandas as pd
from tabulate import tabulate

with open('dumps/ising/tune_mtm_pncg.json') as f:
    mtm_results = json.load(f)

df = pd.DataFrame(mtm_results)

agg_df = df.groupby(['seqlen', 'num_samples', 'alpha']).agg(
    mean_final_tvd=('final_tvd', 'mean'),
    std_final_tvd=('final_tvd', 'std'),
    mean_accept_rate=('accept_rate', 'mean'),
    mean_wallclock=('wallclock', 'mean')
).reset_index()

def format_table(data):
    formatted_data = data.copy()
    formatted_data['alpha'] = formatted_data['alpha'].map(lambda x: f"{x:.4f}")
    formatted_data['mean_final_tvd'] = formatted_data['mean_final_tvd'].map(lambda x: f"{x:.6f}")
    formatted_data['std_final_tvd'] = formatted_data['std_final_tvd'].map(lambda x: f"{x:.6f}")
    formatted_data['mean_accept_rate'] = formatted_data['mean_accept_rate'].map(lambda x: f"{x:.2%}")
    formatted_data['mean_wallclock'] = formatted_data['mean_wallclock'].map(lambda x: f"{x:.4f}s")

    if 'seqlen' in formatted_data.columns:
        formatted_data = formatted_data.drop(columns=['seqlen'])

    return formatted_data

unique_seqlens = sorted(agg_df['seqlen'].unique())

best_settings = []
for seqlen in unique_seqlens:
    seqlen_data = agg_df[agg_df['seqlen'] == seqlen]
    best_row = seqlen_data.loc[seqlen_data['mean_final_tvd'].idxmin()]
    best_settings.append({
        'seqlen': int(best_row['seqlen']),
        'num_samples': int(best_row['num_samples']),
        'alpha': best_row['alpha'],
        'mean_final_tvd': best_row['mean_final_tvd'],
        'std_final_tvd': best_row['std_final_tvd'],
        'mean_accept_rate': best_row['mean_accept_rate'],
        'mean_wallclock': best_row['mean_wallclock'],
    })

best_df = pd.DataFrame(best_settings)
formatted_best = format_table(best_df)
print(tabulate(formatted_best, headers='keys', tablefmt='pipe', showindex=False))

for seqlen in unique_seqlens:
    print(f"\n## Sequence Length = {seqlen}\n")
    seqlen_data = agg_df[agg_df['seqlen'] == seqlen]
    seqlen_data = seqlen_data.sort_values(['num_samples', 'alpha'])
    formatted_data = format_table(seqlen_data)
    print(tabulate(formatted_data, headers='keys', tablefmt='pipe', showindex=False))

# %%
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

with open('dumps/ising/tune_mtm_pncg.json') as f:
    mtm_results = json.load(f)

df = pd.DataFrame(mtm_results)

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})

df = df.groupby(['seqlen', 'num_samples', 'alpha'])

df = df.agg(
    mean_avg_tail_tvd=('avg_tail_tvd', 'mean'),
    std_avg_tail_tvd=('avg_tail_tvd', 'std'),
    mean_final_tvd=('final_tvd', 'mean'),
    std_final_tvd=('final_tvd', 'std'),
    mean_accept_rate=('accept_rate', 'mean'),
    mean_wallclock=('wallclock', 'mean')
).reset_index()

colors = sns.color_palette("viridis", 4)
sample_sizes = [4, 8, 16, 32]

for seqlen in sorted(df['seqlen'].unique()):
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, num_samples in enumerate(sample_sizes):
        subset = df[(df['seqlen'] == seqlen) & (df['num_samples'] == num_samples)]
        subset = subset[subset['alpha'] >= 1.0]
        subset = subset.sort_values('alpha')

        x = subset['alpha']
        y = subset['mean_final_tvd']
        yerr = subset['std_final_tvd']

        ax.plot(x, y, marker='o', color=colors[i], label=f'bsz={num_samples}')
        # ax.fill_between(x, y-yerr, y+yerr, color=colors[i], alpha=0.3)

    # ax.set_xscale('log')
    ax.set_xlabel('Step Size')
    ax.set_ylabel('TVD')
    ax.set_title(f'TVD after 1000 steps, n={seqlen}')
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'figures/tvd_vs_alpha_seqlen_{seqlen}.png', dpi=300)
    # plt.show()

# %%
with open('dumps/ising/tune_pncg.json') as f:
    baseline_results = json.load(f)

df = pd.DataFrame(baseline_results)

# %%
import json
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

with open('dumps/ising/tune_pncg.json') as f:
    baseline = pd.DataFrame(json.load(f))

with open('dumps/ising/tune_mtm_pncg.json') as f:
    mtm = pd.DataFrame(json.load(f))

with open('dumps/ising/tune_iw_mtm_pncg.json') as f:
    iw = pd.DataFrame(json.load(f))

def tab(jwn):
    print(tabulate(jwn, headers='keys', tablefmt='github'))

def agg(df):
    keys = ['seqlen', 'num_samples', 'alpha'] if 'num_samples' in df.keys() else ['seqlen', 'alpha']
    df = df.groupby(keys).agg(
        mean_final_tvd=('final_tvd', 'mean'),
        std_final_tvd=('final_tvd', 'std'),
        mean_accept_rate=('accept_rate', 'mean'),
        mean_wallclock=('wallclock', 'mean')
    ).reset_index()
    best_settings = []
    for seqlen in [4, 8, 16]:
        dfp = df[df['seqlen'] == seqlen]
        best_row = dfp.loc[dfp['mean_final_tvd'].idxmin()]
        best_settings.append({
            'seqlen': int(best_row['seqlen']),
            'alpha': best_row['alpha'],
            'mean_final_tvd': best_row['mean_final_tvd'],
            'std_final_tvd': best_row['std_final_tvd'],
            'mean_accept_rate': best_row['mean_accept_rate'],
            'mean_wallclock': best_row['mean_wallclock'],
        })
        if 'num_samples' in best_row:
            best_settings[-1]['num_samples'] = int(best_row['num_samples']),
    return pd.DataFrame(best_settings)

# %%
import json
import pandas as pd
import matplotlib.pyplot as plt

with open('dumps/ising/tune_pncg.json') as f:
    baseline = pd.DataFrame(json.load(f))

with open('dumps/ising/tune_mtm_pncg.json') as f:
    mtm = pd.DataFrame(json.load(f))

with open('dumps/ising/tune_iw_mtm_pncg.json') as f:
    iw = pd.DataFrame(json.load(f))

for df in [baseline, mtm, iw]:
    df['alpha'] = pd.to_numeric(df['alpha'])

seqlens = [4, 8, 16]
for seqlen in seqlens:
    base_seq = baseline[baseline['seqlen'] == seqlen]
    mtm_seq = mtm[mtm['seqlen'] == seqlen]
    iw_seq = iw[iw['seqlen'] == seqlen]

    plt.figure(figsize=(6, 4))

    base_group = base_seq.groupby('alpha').agg(
        mean_tvd=('final_tvd', 'mean'),
        std_tvd=('final_tvd', 'std')
    ).reset_index()
    plt.plot(base_group['alpha'], base_group['mean_tvd'], 'k-', label='Baseline', linewidth=2)

    for ns in sorted(mtm_seq['num_samples'].unique()):
        mtm_ns = mtm_seq[mtm_seq['num_samples'] == ns]
        mtm_group = mtm_ns.groupby('alpha').agg(
            mean_tvd=('final_tvd', 'mean'),
            std_tvd=('final_tvd', 'std')
        ).reset_index()
        plt.plot(mtm_group['alpha'], mtm_group['mean_tvd'], '-o', label=f'MTM B={int(ns)}')

    plt.xlabel('Step Size')
    plt.ylabel('TVD')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'figures/seqlen_{seqlen}_mtm_vs_baseline.png', dpi=300)

    plt.figure(figsize=(6, 4))

    plt.plot(base_group['alpha'], base_group['mean_tvd'], 'k-', label='Baseline', linewidth=2)

    for ns in sorted(iw_seq['num_samples'].unique()):
        iw_ns = iw_seq[iw_seq['num_samples'] == ns]
        iw_group = iw_ns.groupby('alpha').agg(
            mean_tvd=('final_tvd', 'mean'),
            std_tvd=('final_tvd', 'std')
        ).reset_index()
        plt.plot(iw_group['alpha'], iw_group['mean_tvd'], '-o', label=f'IW-MTM B={int(ns)}')

    plt.xlabel('Step Size')
    plt.ylabel('TVD')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'figures/seqlen_{seqlen}_iw_vs_baseline.png', dpi=300)
