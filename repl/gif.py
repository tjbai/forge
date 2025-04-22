import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from forge.ising import *

SEED = 41
random.seed(SEED)
torch.manual_seed(SEED)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

VOCAB = torch.tensor([[[-1., 1.]]], device=device)
V = VOCAB.shape[-1]

def run_pncg_hist(alpha=1.0, beta=0.42, p=1.0, bsz=1, seqlen=4, steps=200, seed=SEED):
    state = init_state(bsz, seqlen, seed)
    exact = compute_exact_dist(seqlen, beta)
    counts = torch.zeros(2**seqlen, device='cpu')
    emps, tvds, states = [], [], []

    for _ in range(steps):
        E = ncycle_energy(state, beta); E.sum().backward()
        with torch.no_grad():
            pd = pncg_dist(state, alpha, p)
            samp = pncg_sample(pd)[0]
            p_samp = prop_prob(samp.unsqueeze(0), pd)
        samp = samp.unsqueeze(0).detach().requires_grad_(True)
        E_t = ncycle_energy(samp, beta); E_t.sum().backward()
        with torch.no_grad():
            pd2 = pncg_dist(samp, alpha, p)
            p_state = prop_prob(state, pd2)
        acc = mh_accept(state, E, p_state, samp, E_t, p_samp)
        state = torch.where(acc.unsqueeze(1), samp.detach(), state.detach()).requires_grad_(True)

        counts[state_to_index(state)[0]] += 1
        dist = (counts / counts.sum()).cpu().numpy()
        emps.append(dist)
        tvds.append(0.5 * np.abs(exact - dist).sum())
        states.append(state.detach().cpu().numpy()[0])

    return exact, np.array(emps), np.array(tvds), np.array(states)

def make_gif(exact, emps, tvds, states, filename='pncg.gif'):
    steps, _ = emps.shape
    seqlen = states.shape[1]
    x = np.arange(len(exact))

    plt.style.use('ggplot')  # Add modern style
    fig, axes = plt.subplots(1, 4, figsize=(24, 5), facecolor='#f8f8f8')
    fig.tight_layout(pad=3.0)  # Better spacing
    ax_exact, ax_emp, ax_tvd, ax_state = axes.flatten()

    def update(i):
        for ax in axes.flatten():
            ax.clear()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        ax_exact.bar(x, exact, color='#3274A1', alpha=0.8, edgecolor='white', linewidth=0.7)
        ax_exact.set_title('Exact Distribution', fontsize=14, fontweight='bold')
        ax_exact.grid(axis='y', alpha=0.3)

        ax_emp.bar(x, (i+1) * emps[i], color='#E1812C', alpha=0.8, edgecolor='white', linewidth=0.7)
        ax_emp.set_title('Empirical Distribution', fontsize=14, fontweight='bold')
        ax_emp.grid(axis='y', alpha=0.3)

        ax_tvd.plot(tvds[:i+1], color='#3A923A', linewidth=2.5)
        ax_tvd.scatter(i, tvds[i], color='#3A923A', s=100, zorder=5)  # Highlight current point
        ax_tvd.set_ylim(0, 1.0)
        ax_tvd.set_title('TVD', fontsize=14, fontweight='bold')
        ax_tvd.grid(alpha=0.3)

        ax_state.set_xlim(-0.5, seqlen-0.5)
        ax_state.set_ylim(-1.2, 1.2)
        ax_state.set_xticks(range(seqlen))
        ax_state.set_yticks([])

        for j in range(seqlen):
            state_val = states[i][j]
            ax_state.arrow(
                j, 0, 0, 0.7 * (-1 if state_val < 0 else 1), head_width=0.2, head_length=0.2,
                fc='black', ec='black', lw=2.0
            )

        ax_state.set_title('State', fontsize=14, fontweight='bold')

        fig.suptitle(f'Step: {i+1}/{steps}', fontsize=16, y=0.98)

    anim = animation.FuncAnimation(fig, update, frames=steps, interval=100)
    anim.save(filename, writer='pillow', fps=50, dpi=200)

if __name__ == '__main__':
    exact, emps, tvds, states = run_pncg_hist(seqlen=4, steps=1000)
    make_gif(exact, emps, tvds, states)
