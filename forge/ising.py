import torch
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
VOCAB = torch.tensor([[[-1., 1.]]], device=device)

def ncycle_energy(state: torch.Tensor, beta: float):
    return -beta * torch.sum(state * torch.roll(state, shifts=1, dims=1), dim=1) # (B,)

def pncg_dist(state: torch.Tensor, alpha: float = 1.0, p: float = 1.0) -> torch.Tensor:
    assert state.grad is not None
    diffs = VOCAB - state.unsqueeze(-1) # (B, N, V)
    means = (-1/2) * state.grad.unsqueeze(-1) * diffs # (B, N, V)
    regs = (-1/(2*alpha)) * torch.norm(diffs.unsqueeze(-1), p=p, dim=-1) # (B, N, V)
    return F.log_softmax(means + regs, dim=2) # (B, N, V)

def pncg_sample(prop_dist: torch.Tensor) -> torch.Tensor:
    B, N, V = prop_dist.shape
    flat_dist = prop_dist.reshape(-1, V) # (BN, V)
    flat_indices = torch.multinomial(torch.exp(flat_dist), num_samples=1).squeeze(-1) # (BN,)
    return VOCAB.squeeze()[flat_indices.reshape(B, N)] # (B, N)

def prop_prob(state: torch.Tensor, prop_dist: torch.Tensor):
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
    accept_prob = torch.clamp(torch.exp(sample_energy + state_prob - state_energy - sample_prob), max=1)
    return torch.rand_like(accept_prob) < accept_prob

if __name__ == '__main__':
    ALPHA, BETA, P = 1.0, 0.42, 1.0
    B, N = 1, 5

    generator = torch.Generator(device=device)
    generator.manual_seed(42)

    state = 2 * torch.randint(
        0, 2, (B, N),
        device=device,
        dtype=torch.float32,
        generator=generator,
    ) - 1
    state.requires_grad_(True)

    total_accepted = 0
    for i in range(500):
        state_energy = ncycle_energy(state, beta=BETA)
        state_energy.sum().backward()

        with torch.no_grad():
            prop_dist = pncg_dist(state, alpha=ALPHA, p=P)
            sample = pncg_sample(prop_dist)
            sample_prob = prop_prob(sample, prop_dist)

        sample = sample.detach().clone().requires_grad_(True)
        sample_energy = ncycle_energy(sample, beta=BETA)
        sample_energy.sum().backward()

        with torch.no_grad():
            prop_dist = pncg_dist(sample, alpha=ALPHA, p=P)
            state_prob = prop_prob(state, prop_dist)

        accept_index = mh_accept(
            state, state_energy, state_prob,
            sample, sample_energy, sample_prob
        )
        total_accepted += accept_index.sum()

        with torch.no_grad():
            state = torch.where(
                accept_index.unsqueeze(1),
                sample.detach().clone(),
                state.detach().clone()
            )
            state.requires_grad_(True)

        if (i+1) % 100 == 0:
            print('energy', state_energy)
            print('acceptance rate', total_accepted / ((i+1) * B))
