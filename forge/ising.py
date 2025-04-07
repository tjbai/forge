from typing import Dict

import torch
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
VOCAB = torch.tensor([[[-1., 1.]]], device=device)

def energy(state: torch.Tensor, beta: float):
    return -beta * torch.sum(state * torch.roll(state, shifts=1, dims=1))

def pncg_dist(
    state: torch.Tensor,
    alpha: float = 1.0,
    p: float = 1.0,
) -> torch.Tensor:
    assert state.grad is not None
    B, N = state.shape
    diffs = VOCAB - state.unsqueeze(-1)
    means = (-1/2) * state.grad.unsqueeze(-1) * diffs
    regs = (-1/(2*alpha)) * torch.norm(diffs.unsqueeze(-1), p=p, dim=-1)
    return F.log_softmax(means + regs, dim=2)

def pncg_sample(state: torch.Tensor, **prop_args) -> Dict:
    B, N = state.shape
    probs = pncg_dist(state, **prop_args)
    flat_probs = probs.reshape(-1, 2)
    flat_indices = torch.multinomial(torch.exp(flat_probs), num_samples=1).squeeze(-1)
    sample_probs = flat_probs[torch.arange(B*N), flat_indices.squeeze()].reshape(B, N)
    return {
        'sample': VOCAB.squeeze()[flat_indices.reshape(B, N)],
        'sample_prob': torch.sum(sample_probs, dim=-1)
    }

def mh(
    state: torch.Tensor,
    state_energy: torch.Tensor,
    sample: torch.Tensor,
    sample_prob: torch.Tensor,
) -> bool:
    # we have p(sample | state) and E(state)
    # we need p(state | sample) and E(sample)
    # to get the latter we can energy(sample).backward()
    sample_energy = energy(sample)

    return True

'''
ALPHA = 1.0
P = 1.0

# everything deals with an extra batch dimension that we're going to deal with later
state = torch.tensor([[-1., 1. ,-1.]], requires_grad=True)
state_energy = energy(state, beta=0.42)
state_energy.backward() # this populates state.grad

prop = pncg_sample(state, alpha=ALPHA, p=P)

state = sample['sample_state'] if mh(
    state,
    state_energy,
    prop['sample'][0],
    prop['sample_prob'][0]
) else state
'''
