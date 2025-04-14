import time
from typing import Optional

import torch
import torch.nn.functional as F
import fire
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

MODEL_NAME = 'openai-community/gpt2'
device = 'cuda' if torch.cuda.is_available() else 'mps'

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device).eval()

VOCAB_SIZE = model.config.vocab_size
EMBEDDING_DIM = model.config.hidden_size
EMBEDDINGS = model.get_input_embeddings().weight.detach() # (V, E)

def get_embeddings(token_ids: torch.Tensor):
    return model.get_input_embeddings()(token_ids)

def lm_energy(
    state: torch.Tensor,
    state_embeddings: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    B, N, E = state_embeddings.shape
    outputs = model(inputs_embeds=state_embeddings, attention_mask=attn_mask, return_dict=True)
    logits = outputs.logits # (B, N, V)

    # TODO -- think about EOS here
    dists = logits[:, :-1, :].contiguous()      # (B, N-1, V)
    labels = state[:, 1:].contiguous()          # (B, N-1)
    neg_log_probs = F.cross_entropy(            # (B,)
        dists.view(-1, VOCAB_SIZE),
        labels.view(-1),
        reduction='none',
    ).view(B, N-1).sum(dim=1)

    return neg_log_probs

def pncg_dist(
    state_embeddings: torch.Tensor, # (B, N, E)
    gradients: torch.Tensor,        # (B, N, E)
    alpha: float = 1.0,
    p: float = 1.0,
):
    diffs = (
        EMBEDDINGS.view(1, 1, VOCAB_SIZE, EMBEDDING_DIM) # (1, 1, V, E)
        - state_embeddings.unsqueeze(2)                  # (B, N, E) -> (B, N, 1, E)
    )                                                    # (B, N, V, E)

    means = -1/2 * torch.einsum( # (B, N, V)
        'bnve,bnke->bnv',
        gradients.unsqueeze(2),  # (B, N, 1, E)
        diffs                    # (B, N, V, E)
    )

    # (B, N, V)
    if p == 1.0:
        norms = -1/(2*alpha) * torch.sum(torch.abs(diffs), dim=-1)
    elif p == 2.0:
        norms = -1/(2*alpha) * torch.sqrt(torch.sum(diffs**2, dim=-1))
    else:
        norms = -1/(2*alpha) * torch.norm(diffs, p=p, dim=-1)

    return F.log_softmax(means + norms, dim=-1) # (B, N, V)

def pncg_sample(prop_dist: torch.Tensor, k: int = 1):
    B, N, _ = prop_dist.shape
    flat_dist = prop_dist.reshape(-1, VOCAB_SIZE) # (BN, V)
    if k == 1:
        flat_indices = torch.multinomial(torch.exp(flat_dist), num_samples=1).squeeze(-1) # (BN,)
        return flat_indices.reshape(B, N)
    flat_indices = torch.multinomial(torch.exp(flat_dist), num_samples=k, replacement=True) # (BN, K)
    return flat_indices.view(B, N, k).permute(0, 2, 1) # (B, N, K) -> (B, K, N)

def prop_prob(
    state: torch.Tensor,    # (B_1, N)
    prop_dist: torch.Tensor # (B_2, N, V)
):
    B_2, *_ = prop_dist.shape
    state = state.unsqueeze(1).unsqueeze(-1).expand(-1, B_2, -1, -1) # (B_1, B_2, N, 1)
    prop_dist = prop_dist.unsqueeze(0)                               # (1, B_2, N, V)
    log_probs = torch.gather(prop_dist, dim=3, index=state)          # (B_1, B_2, N, 1)
    return torch.sum(log_probs, dim=2).squeeze(2)                    # (B_1, B_2)

def init_pncg_state(bsz: int, seqlen: int, seed: int):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    state_ids = torch.randint(0, VOCAB_SIZE, (bsz, seqlen), device=device, generator=generator)
    # state_ids[:, 0] = tokenizer.bos_token_id # NOTE -- gpt-2 doesn't have bos token
    return state_ids

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
    wandb.init(
        project='mcmc',
        config={
            'method': 'mtm_pncg',
            'alpha': alpha,
            'beta': beta,
            'p': p,
            'seqlen': seqlen,
            'steps': steps,
            'seed': seed,
            'num_samples': num_samples,
        }
    )

    x = init_pncg_state(1, seqlen, seed)

    s = time.time()
    energies = []
    states = []
    total_accepted = 0
    for i in tqdm(range(steps), disable=quiet):
        x_embeds = get_embeddings(x).detach().clone().requires_grad_(True)
        x_energy = lm_energy(x, x_embeds)
        x_energy.sum().backward()

        energies.append(x_energy.item())
        states.append(x.squeeze().tolist())
        wandb.log({'energy': energies[-1]})

        with torch.no_grad():
            prop_dist = pncg_dist(x_embeds, x_embeds.grad, alpha=alpha, p=p)
            ys = pncg_sample(prop_dist, k=num_samples).squeeze(0) # (K, N)

        ys_embeds = get_embeddings(ys).detach().clone().requires_grad_(True)
        y_energies = lm_energy(ys, ys_embeds)
        y_energies.sum().backward()

        with torch.no_grad():
            prop_dist = pncg_dist(ys_embeds, ys_embeds.grad, alpha=alpha, p=p) # (K, N, V)
            txy = prop_prob(x, prop_dist).squeeze(0) # (K,)
            log_forward_weights = txy - y_energies

        selected_idx = torch.multinomial(
            F.softmax(log_forward_weights, dim=0),
            num_samples=1
        ).squeeze()

        with torch.no_grad():
            yk_prop_dist = pncg_dist(
                ys_embeds[selected_idx].unsqueeze(0),
                ys_embeds.grad[selected_idx].unsqueeze(0),
                alpha=alpha, p=p
            )
            ref_xs = pncg_sample(yk_prop_dist, k=num_samples-1).squeeze(0) # (K-1, N)

        ref_set = torch.cat((x, ref_xs), dim=0)
        ref_set_embeds = get_embeddings(ref_set).detach().clone().requires_grad_(True)
        ref_set_energies = lm_energy(ref_set, ref_set_embeds)
        ref_set_energies.sum().backward()

        with torch.no_grad():
            prop_dist = pncg_dist(ref_set_embeds, ref_set_embeds.grad, alpha=alpha, p=p)
            tyx = prop_prob(ys[selected_idx].unsqueeze(0), prop_dist).squeeze(0)
            log_reverse_weights = tyx - ref_set_energies

        log_accept_ratio = torch.logsumexp(log_forward_weights, dim=0) - torch.logsumexp(log_reverse_weights, dim=0)
        accept_prob = torch.clamp(torch.exp(log_accept_ratio), max=1.0)

        if torch.rand(1, device=device) < accept_prob:
            x = ys[selected_idx].unsqueeze(0).detach().clone()
            total_accepted += 1

    return {
        'states': states,
        'energies': energies,
        'wallclock': time.time() - s,
        'accept_rate': total_accepted / steps,
    }

def run_pncg(
    alpha: float = 4.0,
    beta: float = 1.0,
    p: float = 1.0,
    bsz: int = 1,
    seqlen: int = 5,
    steps: int = 500,
    seed: int = 42,
    quiet: bool = False,
):
    wandb.init(
        project='mcmc',
        config={
            'method': 'pncg',
            'alpha': alpha,
            'beta': beta,
            'p': p,
            'seqlen': seqlen,
            'steps': steps,
            'seed': seed
        }
    )

    state = init_pncg_state(bsz, seqlen, seed)

    s = time.time()
    energies = []
    states = []
    total_accepted = 0
    for i in tqdm(range(steps), disable=quiet):
        state_embeds = get_embeddings(state).detach().clone().requires_grad_(True)
        state_energy = lm_energy(state, state_embeds)
        state_energy.sum().backward()
        state_grad = state_embeds.grad

        energies.append(state_energy.item())
        states.append(state.squeeze().tolist())
        wandb.log({'energy': energies[-1]})

        with torch.no_grad():
            prop_dist_forward = pncg_dist(state_embeds, state_grad, alpha=alpha, p=p)
            samples = pncg_sample(prop_dist_forward)
            log_prob_forward = prop_prob(samples, prop_dist_forward)
            log_prob_forward = torch.diag(log_prob_forward)

        sample_embeds = get_embeddings(samples).detach().clone().requires_grad_(True)
        sample_energy = lm_energy(samples, sample_embeds)
        sample_energy.sum().backward()
        sample_grad = sample_embeds.grad.detach().clone()

        with torch.no_grad():
            prop_dist_reverse = pncg_dist(sample_embeds, sample_grad, alpha=alpha, p=p)
            log_prob_reverse = prop_prob(state, prop_dist_reverse)
            log_prob_reverse = torch.diag(log_prob_reverse)

        accept_prob = torch.clamp(torch.exp(state_energy - sample_energy + log_prob_reverse - log_prob_forward), max=1)
        accept = torch.rand_like(accept_prob) < accept_prob
        total_accepted += accept.sum().item()

        with torch.no_grad():
            state = torch.where(accept.unsqueeze(1), samples, state)

    wallclock = time.time() - s
    accept_rate = total_accepted / (steps * bsz)

    return {
        'states': states,
        'energies': energies,
        'wallclock': wallclock,
        'accept_rate': accept_rate,
    }

if __name__ == '__main__':
    fire.Fire(run_mtm_pncg)
    # fire.Fire(run_pncg)
