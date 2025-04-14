import time
from typing import List
from itertools import product
from functools import lru_cache

import torch
import torch.nn.functional as F
import fire
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# TODO -- reproducible preamble

device = 'cuda' if torch.cuda.is_available() else 'mps'
VOCAB = torch.tensor([[[-1., 1.]]], device=device)
V = VOCAB.shape[-1]

def ncycle_energy(state: torch.Tensor, beta: float):
    return -beta/2 * torch.sum(state * torch.roll(state, shifts=1, dims=1), dim=1) # (B,)

def pncg_dist(state: torch.Tensor, alpha: float = 1.0, p: float = 1.0) -> torch.Tensor:
    assert state.grad is not None
    diffs = VOCAB - state.unsqueeze(-1) # (B, N, V)
    means = -1/2 * state.grad.unsqueeze(-1) * diffs # (B, N, V)
    regs = -1/(2*alpha) * torch.norm(diffs.unsqueeze(-1), p=p, dim=-1) # (B, N, V)
    return F.log_softmax(means + regs, dim=2) # (B, N, V)

def pncg_sample(prop_dist: torch.Tensor, k: int = 1) -> torch.Tensor:
    B, N, _ = prop_dist.shape
    flat_dist = prop_dist.reshape(-1, V) # (BN, V)
    if k == 1:
        flat_indices = torch.multinomial(torch.exp(flat_dist), num_samples=1).squeeze(-1) # (BN,)
        return VOCAB.squeeze()[flat_indices.reshape(B, N)] # (B, N)
    else:
        flat_indices = torch.multinomial(torch.exp(flat_dist), num_samples=k, replacement=True) # (BN, K)
        flat_indices = flat_indices.view(B, N, k).permute(0, 2, 1) # (B, N, K) -> (B, K, N)
        return VOCAB.squeeze()[flat_indices] # (B, K, N)

def prop_prob(state: torch.Tensor, prop_dist: torch.Tensor):
    if prop_dist.shape[0] < state.shape[0]:
        prop_dist = prop_dist.expand(state.shape[0], *prop_dist.shape[1:])
    elif prop_dist.shape[0] > state.shape[0]:
        state = state.expand(prop_dist.shape[0], *state.shape[1:])
    index = (state.unsqueeze(-1) == VOCAB).long().argmax(dim=-1, keepdim=True) # (B, N, 1)
    probs =  torch.gather(prop_dist, dim=2, index=index) # (B, N)
    return torch.sum(probs, dim=1).squeeze(-1) # (B,)

def mh_accept(
    state: torch.Tensor,
    state_energy: torch.Tensor,
    state_prob: torch.Tensor,
    sample: torch.Tensor,
    sample_energy: torch.Tensor,
    sample_prob: torch.Tensor,
) -> torch.Tensor:
    accept_prob = torch.clamp(torch.exp(state_energy + state_prob - sample_energy - sample_prob), max=1)
    return torch.rand_like(accept_prob) < accept_prob

def state_to_index(state: torch.Tensor) -> List[int]:
    B, N = state.shape
    base = VOCAB.shape[-1]
    acc = torch.zeros(B, device=device, dtype=torch.long)
    index = (state.unsqueeze(-1) == VOCAB).long().argmax(dim=-1, keepdim=False) # (B, N)
    for i in range(N):
        acc += index[:, i] * (base ** i)
    return acc.tolist()

@lru_cache(maxsize=None)
def compute_exact_dist(n: int, beta: float):
    energy = torch.zeros(2**n, dtype=torch.float32, device=device)
    for _state in tqdm(product(VOCAB.squeeze().tolist(), repeat=n), desc='computing exact distribution', total=2**n):
        state = torch.tensor([_state], device=device)
        energy[state_to_index(state)[0]] = ncycle_energy(state, beta=beta)[0].item()
    return F.softmax(-energy, dim=0)

def tvd(exact_dist: torch.Tensor, empirical_dist: torch.Tensor):
    return 1/2 * torch.sum(torch.abs(exact_dist - empirical_dist)).item()

def init_state(bsz: int, seqlen: int, seed: int):
    generator = torch.Generator(device=device)
    generator.manual_seed(42)
    state =  torch.randint(
        0, 2, (bsz, seqlen),
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    state = 2 * state - 1 # (0, 1) -> (-1, 1)
    state.requires_grad_(True)
    return state

def plot_run(
    exact_dist: torch.Tensor,
    empirical_dist: torch.Tensor,
    tvds: List[int],
    wallclock: float,
    steps: int,
    seqlen: int,
    total_accepted: int,
):
    sns.set_theme(style='whitegrid')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

    indices = np.arange(len(exact_dist))
    ax1.plot(indices, exact_dist, linewidth=0.5)
    ax1.fill_between(indices, 0, exact_dist, alpha=0.3)
    ax1.set_title('Exact Distribution')

    ax2.plot(indices, empirical_dist, linewidth=0.5)
    ax2.fill_between(indices, 0, empirical_dist, alpha=0.3)
    ax2.set_title('Empirical Distribution')

    ax3.plot(np.arange(steps), tvds)
    ax3.grid(axis='y', linestyle='--', alpha=0.7)
    ax3.set_ylim(0, 0.5)
    ax3.set_yticks(np.arange(0, 0.5 + 0.05, 0.05))
    ax3.set_title('TVD')

    fig.text(0.05, 0.05, f'n={seqlen}, steps={steps}, runtime={wallclock:.3f}s, sps={steps / wallclock:.3f}, accepted={total_accepted / steps:.2f}')
    plt.tight_layout(pad=3)
    plt.show()

def run_mtm_pncg(
    alpha: float = 1.0,
    beta: float = 0.42,
    p: float = 1.0,
    num_samples: int = 4,
    seqlen: int = 5,
    steps: int = 500,
    seed: int = 42,
    quiet: bool = False
):
    assert num_samples > 1
    x = init_state(1, seqlen, seed) # too lazy to do bsz > 1 and num_samples > 1

    total_accepted = 0
    exact_dist = compute_exact_dist(seqlen, beta)
    empirical_dist = torch.zeros(2**seqlen, dtype=torch.float32, device=device)
    tvds = []

    s = time.time()
    for i in tqdm(range(steps), disable=quiet):
        assert x.grad is None
        x_energy = ncycle_energy(x, beta=beta)
        x_energy.sum().backward()

        with torch.no_grad():
            prop_dist = pncg_dist(x, alpha=alpha, p=p)
            ys = pncg_sample(prop_dist, k=num_samples).squeeze(0) # (K, N)

        ys = ys.detach().clone().requires_grad_(True)
        y_energies = ncycle_energy(ys, beta=beta) # (K,)
        y_energies.sum().backward()

        with torch.no_grad():
            prop_dist = pncg_dist(ys, alpha=alpha, p=p) # (K, N, V)
            txy = prop_prob(x, prop_dist) # (K,)
            log_forward_weights = txy - y_energies

        selected_idx = torch.multinomial(
            F.softmax(log_forward_weights, dim=0),
            num_samples=1
        ).item()

        yk = ys[selected_idx].unsqueeze(0).detach().clone().requires_grad_(True) # (1, N)
        yk.grad = ys.grad[selected_idx].unsqueeze(0).detach().clone()

        with torch.no_grad():
            yk_prop_dist = pncg_dist(yk, alpha=alpha, p=p)
            ref_xs = pncg_sample(yk_prop_dist, k=num_samples-1).squeeze(0) # (K-1, N)

        ref_set = torch.cat((x.detach(), ref_xs), dim=0).requires_grad_(True)
        ref_set_energies = ncycle_energy(ref_set, beta=beta)
        ref_set_energies.sum().backward()

        with torch.no_grad():
            prop_dist = pncg_dist(ref_set, alpha=alpha, p=p)
            tyx = prop_prob(yk, prop_dist)
            log_reverse_weights = tyx - ref_set_energies

        log_accept_ratio = torch.logsumexp(log_forward_weights, dim=0) - torch.logsumexp(log_reverse_weights, dim=0)
        accept_prob = torch.clamp(torch.exp(log_accept_ratio), max=1.0)

        if torch.rand(1, device=device) < accept_prob:
            x = yk.detach().clone().requires_grad_(True)
            total_accepted += 1
        else:
            x = x.detach().clone().requires_grad_(True)

        empirical_dist[state_to_index(x)[0]] += 1
        tvds.append(tvd(exact_dist, empirical_dist / empirical_dist.sum()))

    return {
        'tvds': tvds,
        'empirical_dist': empirical_dist,
        'wallclock': time.time() - s,
        'accept_rate': total_accepted / steps,
    }

def run_pncg(
    alpha: float = 1.0,
    beta: float = 0.42,
    p: float = 1.0,
    bsz: int = 1,
    seqlen: int = 5,
    steps: int = 500,
    seed: int = 42,
    quiet: bool = False,
):
    state = init_state(bsz, seqlen, 42)

    total_accepted = 0
    exact_dist = compute_exact_dist(seqlen, beta)
    empirical_dist = torch.zeros(2**seqlen, dtype=torch.float32, device=device)
    tvds = []

    s = time.time()
    for i in tqdm(range(steps), disable=quiet):
        assert state.grad is None
        state_energy = ncycle_energy(state, beta=beta)
        state_energy.sum().backward()

        with torch.no_grad():
            prop_dist = pncg_dist(state, alpha=alpha, p=p)
            sample = pncg_sample(prop_dist)
            sample_prob = prop_prob(sample, prop_dist)

        sample = sample.detach().clone().requires_grad_(True)
        sample_energy = ncycle_energy(sample, beta=beta)
        sample_energy.sum().backward()

        with torch.no_grad():
            prop_dist = pncg_dist(sample, alpha=alpha, p=p)
            state_prob = prop_prob(state, prop_dist)

        accept_index = mh_accept(
            state, state_energy, state_prob,
            sample, sample_energy, sample_prob
        )
        total_accepted += accept_index.sum().item()

        with torch.no_grad():
            state = torch.where(
                accept_index.unsqueeze(1),
                sample.detach().clone(),
                state.detach().clone()
            )
            state.requires_grad_(True)
            assert state.grad is None

        empirical_dist[state_to_index(state)[0]] += 1
        tvds.append(tvd(exact_dist, empirical_dist / empirical_dist.sum()))

    return {
        'tvds': tvds,
        'empirical_dist': empirical_dist,
        'wallclock': time.time() - s,
        'accept_rate': total_accepted / steps,
    }

if __name__ == '__main__':
    # fire.Fire(run_pncg)
    fire.Fire(run_mtm_pncg)
