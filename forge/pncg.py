import time
import random
from typing import Optional

import torch
import torch.nn.functional as F
import fire
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'mps'

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def log(d):
    wandb.log(d) if wandb.run is not None else print(d)

def get_embeddings(model, token_ids: torch.Tensor):
    return model.get_input_embeddings()(token_ids)

def lm_energy(
    model: AutoModelForCausalLM,
    state: torch.Tensor,
    state_embeddings: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    beta: float = 1.0,
) -> torch.Tensor:
    B, N, E = state_embeddings.shape
    outputs = model( # type: ignore
        inputs_embeds=state_embeddings,
        attention_mask=attn_mask,
        return_dict=True
    )
    logits = outputs.logits # (B, N, V)

    # TODO -- think about EOS here
    dists = logits[:, :-1, :].contiguous()      # (B, N-1, V)
    labels = state[:, 1:].contiguous()          # (B, N-1)
    neg_log_probs = F.cross_entropy(            # (B,)
        dists.view(-1, model.config.vocab_size),
        labels.view(-1),
        reduction='none',
    ).view(B, N-1).sum(dim=1)

    return neg_log_probs
    # return beta * neg_log_probs

def pncg_dist(
    embeddings: torch.Tensor,       # (V, E)
    state_embeddings: torch.Tensor, # (B, N, E)
    gradients: torch.Tensor,        # (B, N, E)
    alpha: float = 1.0,
    p: float = 1.0,
):
    V, E, = embeddings.shape
    diffs = (
        embeddings.view(1, 1, V, E)     # (1, 1, V, E)
        - state_embeddings.unsqueeze(2) # (B, N, E) -> (B, N, 1, E)
    )                                   # (B, N, V, E)

    means = -1/2 * torch.einsum( # (B, N, V)
        'bnve,bnke->bnv',
        diffs,                   # (B, N, V, E)
        gradients.unsqueeze(2),  # (B, N, 1, E)
    )

    # (B, N, V)
    if p == 1.0:
        norms = -1/(2*alpha) * torch.sum(torch.abs(diffs), dim=-1)
    elif p == 2.0:
        norms = -1/(2*alpha) * torch.sqrt(torch.sum(diffs**2, dim=-1))
    else:
        norms = -1/(2*alpha) * torch.norm(diffs, p=p, dim=-1)

    return F.log_softmax(means + norms, dim=-1) # (B, N, V)

def pncg_dist_p2(
    embeddings: torch.Tensor,
    state_embeddings: torch.Tensor,
    gradients: torch.Tensor,
    alpha: float = 1.0,
    **_,
):
    # memory-efficient implementation of pncg_dist for p=2
    B, N, E = state_embeddings.shape
    V, _ = embeddings.shape

    grads_flat = gradients.view(-1, E)
    term1 = (grads_flat @ embeddings.T).view(B, N, V)
    term2 = (gradients * state_embeddings).sum(dim=-1).unsqueeze(-1)
    means = -1/2 * (term1 - term2)

    state_emb_flat = state_embeddings.view(-1, E)
    dot_prod = (state_emb_flat @ embeddings.T).view(B, N, V)
    state_emb_norm_sq = (state_embeddings**2).sum(dim=-1).unsqueeze(-1)
    emb_v_norm_sq = (embeddings**2).sum(dim=-1).view(1, 1, V)
    dist_sq = emb_v_norm_sq + state_emb_norm_sq - 2 * dot_prod
    dist_sq = torch.clamp(dist_sq, min=1e-9)
    norms = -1/(2*alpha) * torch.sqrt(dist_sq)

    return F.log_softmax(means + norms, dim=-1) # (B, N, V)

def pncg_sample(prop_dist: torch.Tensor, k: int = 1):
    B, N, V = prop_dist.shape
    flat_dist = prop_dist.reshape(-1, V) # (BN, V)
    if k == 1:
        flat_indices = torch.multinomial(torch.exp(flat_dist), num_samples=1).squeeze(-1) # (BN,)
        return flat_indices.reshape(B, N)
    flat_indices = torch.multinomial(torch.exp(flat_dist), num_samples=k, replacement=True) # (BN, K)
    return flat_indices.view(B, N, k).permute(0, 2, 1) # (B, N, K) -> (B, K, N)

def prop_prob(
    state: torch.Tensor,    # (B_1, N)
    prop_dist: torch.Tensor # (B_2, N, V)
):
    (B_1, _), (B_2, *_) = state.shape, prop_dist.shape
    state = state.unsqueeze(1).unsqueeze(-1).expand(-1, B_2, -1, -1) # (B_1, B_2, N, 1)
    prop_dist = prop_dist.unsqueeze(0).expand(B_1, -1, -1, -1)       # (B_1, B_2, N, V)
    log_probs = torch.gather(prop_dist, dim=3, index=state)          # (B_1, B_2, N, 1)
    return torch.sum(log_probs, dim=2).squeeze(2)                    # (B_1, B_2)

def init_pncg_state(bsz: int, seqlen: int, seed: int, vocab_size: int):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    state_ids = torch.randint(0, vocab_size, (bsz, seqlen), device=device, generator=generator)
    # state_ids[:, 0] = tokenizer.bos_token_id # NOTE -- gpt-2 doesn't have bos token
    return state_ids

def run_mtm_pncg(
    model_name: str = 'openai-community/gpt2',
    alpha: float = 1.0,
    beta: float = 0.42,
    p: float = 1.0,
    num_samples: int = 4,
    seqlen: int = 5,
    steps: int = 500,
    seed: int = 42,
    quiet: bool = False,
    init_wandb: bool = False,
    run_name: str = 'mtm_pncg',
    ema_lambda: float = 0.0,
):
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
    embeddings = model.get_input_embeddings().weight.detach()
    vocab_size = embeddings.shape[0]

    if init_wandb:
        wandb.init(
            project='mcmc',
            name=run_name,
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

    x = init_pncg_state(1, seqlen, seed, vocab_size)

    s = time.time()
    energies = []
    ema_energy = None
    states = []
    total_accepted = 0
    for i in tqdm(range(steps), disable=quiet):
        x_embeds = get_embeddings(model, x).detach().clone().requires_grad_(True)
        x_energy = lm_energy(model, x, x_embeds, beta=beta)
        x_energy.sum().backward()

        energies.append(x_energy.item())
        states.append(x.squeeze().tolist())
        ema_energy = (
            energies[-1] if ema_energy is None else
            ema_lambda * ema_energy + (1 - ema_lambda) * energies[-1]
        )
        log({'energy': energies[-1], 'ema_energy': ema_energy})

        with torch.no_grad():
            prop_dist = pncg_dist_p2(embeddings, x_embeds, x_embeds.grad, alpha=alpha, p=p)
            ys = pncg_sample(prop_dist, k=num_samples).squeeze(0) # (K, N)

        ys_embeds = get_embeddings(model, ys).detach().clone().requires_grad_(True)
        y_energies = lm_energy(model, ys, ys_embeds, beta=beta)
        y_energies.sum().backward()

        with torch.no_grad():
            prop_dist = pncg_dist_p2(embeddings, ys_embeds, ys_embeds.grad, alpha=alpha, p=p) # (K, N, V)
            txy = prop_prob(x, prop_dist).squeeze(0) # (K,)
            log_forward_weights = txy - y_energies

        selected_idx = torch.multinomial(
            F.softmax(log_forward_weights, dim=0),
            num_samples=1
        ).squeeze()

        with torch.no_grad():
            yk_prop_dist = pncg_dist_p2(
                embeddings,
                ys_embeds[selected_idx].unsqueeze(0),
                ys_embeds.grad[selected_idx].unsqueeze(0),
                alpha=alpha, p=p
            )
            ref_xs = pncg_sample(yk_prop_dist, k=num_samples-1).squeeze(0) # (K-1, N)

        ref_set = torch.cat((x, ref_xs), dim=0)
        ref_set_embeds = get_embeddings(model, ref_set).detach().clone().requires_grad_(True)
        ref_set_energies = lm_energy(model, ref_set, ref_set_embeds, beta=beta)
        ref_set_energies.sum().backward()

        with torch.no_grad():
            prop_dist = pncg_dist_p2(embeddings, ref_set_embeds, ref_set_embeds.grad, alpha=alpha, p=p)
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
    model_name: str = 'openai-community/gpt2',
    alpha: float = 4.0,
    beta: float = 1.0,
    p: float = 1.0,
    bsz: int = 1,
    seqlen: int = 5,
    steps: int = 500,
    seed: int = 42,
    quiet: bool = False,
    init_wandb: bool = False,
    run_name: str = 'pncg',
    ema_lambda: float = 0.0,
):
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
    embeddings = model.get_input_embeddings().weight.detach()
    vocab_size = embeddings.shape[0]

    if init_wandb:
        wandb.init(
            project='mcmc',
            name=run_name,
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

    state = init_pncg_state(bsz, seqlen, seed, vocab_size)

    s = time.time()
    energies = []
    ema_energy = None
    states = []
    total_accepted = 0
    for i in tqdm(range(steps), disable=quiet):
        state_embeds = get_embeddings(model, state).detach().clone().requires_grad_(True)
        state_energy = lm_energy(model, state, state_embeds, beta=beta)
        state_energy.sum().backward()
        state_grad = state_embeds.grad

        energies.append(state_energy.item())
        states.append(state.squeeze().tolist())
        ema_energy = (
            energies[-1] if ema_energy is None else
            ema_lambda * ema_energy + (1 - ema_lambda) * energies[-1]
        )
        log({'energy': energies[-1], 'ema_energy': ema_energy})

        with torch.no_grad():
            prop_dist_forward = pncg_dist(embeddings, state_embeds, state_grad, alpha=alpha, p=p)
            samples = pncg_sample(prop_dist_forward)
            log_prob_forward = prop_prob(samples, prop_dist_forward)
            log_prob_forward = torch.diag(log_prob_forward)

        sample_embeds = get_embeddings(model, samples).detach().clone().requires_grad_(True)
        sample_energy = lm_energy(model, samples, sample_embeds)
        sample_energy.sum().backward()
        sample_grad = sample_embeds.grad.detach().clone()

        with torch.no_grad():
            prop_dist_reverse = pncg_dist(embeddings, sample_embeds, sample_grad, alpha=alpha, p=p)
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
    fire.Fire({
        'pncg': run_pncg,
        'mtm_pncg': run_mtm_pncg,
