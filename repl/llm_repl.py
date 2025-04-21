# %%
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

with open('dumps/llm/best_pncg.json') as f:
    baseline = json.load(f)

with open('dumps/llm/best_mtm_pncg.json') as f:
    mtm = json.load(f)

with open('dumps/llm/best_iw_mtm_pncg.json') as f:
    iw = json.load(f)

plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")

colors = {
    'baseline': '#3366CC',
    'mtm': '#FF9900',
    'iw': '#33AA33'
}

def plot_method(jwn, label=None, color=None):
    wallclock = np.mean([d['wallclock'] for d in jwn])
    energies = list(zip(*[d['energies'] for d in jwn]))
    means = [np.mean(d) for d in energies]
    lower = [np.percentile(d, q=[5])[0] for d in energies]
    upper = [np.percentile(d, q=[95])[0] for d in energies]

    cross = next((i for i, e in enumerate(means) if e <= 175), 5000)
    cross_time = (cross / 5000) * wallclock

    xs = np.arange(5000)
    plt.plot(xs, means, label=label, color=color, linewidth=2)
    plt.fill_between(xs, lower, upper, alpha=0.2, color=color)

    plt.axvline(cross, color=color, linestyle='--', alpha=0.7)

    text_y = 240 if label == "Importance Weighted" else (260 if label == "Multiple-Try" else 280)
    plt.annotate(
        f"{label}: ({cross_time:.0f} sec)",
        xy=(cross, 175),
        xytext=(cross+100, text_y),
        color=color,
        fontweight='bold',
        arrowprops=dict(arrowstyle='->', color=color, alpha=0.7)
    )

    return cross, cross_time

baseline_cross, baseline_time = plot_method(baseline, label="Baseline", color=colors['baseline'])
mtm_cross, mtm_time = plot_method(mtm, label="Multiple-Try", color=colors['mtm'])
iw_cross, iw_time = plot_method(iw, label="Importance Weighted", color=colors['iw'])

plt.axhline(175, color='#777777', linestyle='-', alpha=0.8)

plt.ylabel('Energy $-\\log p(x)$', fontsize=14)
plt.xlabel('Step', fontsize=14)
plt.xlim(0, 5000)
plt.ylim(100, 300)
plt.grid(True, alpha=0.5)
plt.legend(loc='lower right', framealpha=0.8)
plt.tight_layout()

# plt.show()
plt.savefig('figures/llm_comp.png', dpi=300)
