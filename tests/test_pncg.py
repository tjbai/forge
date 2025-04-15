import pytest
import torch
import torch.nn.functional as F
from forge.pncg import (
    lm_energy,
    pncg_dist,
    pncg_dist_p2,
    pncg_sample,
    prop_prob,
)

TEST_VOCAB_SIZE = 100
TEST_EMBEDDING_DIM = 16
TEST_SEQ_LEN = 5
TEST_BATCH_SIZE = 2
TEST_K = 3

@pytest.fixture(scope="module")
def test_embeddings_tensor():
    return torch.randn(TEST_VOCAB_SIZE, TEST_EMBEDDING_DIM, device='cpu') * 0.1

@pytest.fixture(scope="module")
def mock_model(test_embeddings_tensor):
    class MockGPT2:
        def __init__(self, embeddings):
            self.embeddings = torch.nn.Embedding.from_pretrained(embeddings, freeze=True)
            self.lm_head = torch.nn.Linear(TEST_EMBEDDING_DIM, TEST_VOCAB_SIZE, bias=False)
            torch.nn.init.xavier_uniform_(self.lm_head.weight)

        def get_input_embeddings(self):
            return self.embeddings

        def __call__(self, inputs_embeds, attention_mask=None, return_dict=True):
            logits = self.lm_head(inputs_embeds)
            class MockOutput:
                def __init__(self, logits):
                    self.logits = logits
            return MockOutput(logits=logits)

        @property
        def config(self):
            class MockConfig:
                vocab_size = TEST_VOCAB_SIZE
                hidden_size = TEST_EMBEDDING_DIM
            return MockConfig()
    return MockGPT2(test_embeddings_tensor)

@pytest.fixture
def sample_state():
    return torch.randint(0, TEST_VOCAB_SIZE, (TEST_BATCH_SIZE, TEST_SEQ_LEN), device='cpu')

@pytest.fixture
def sample_state_embeddings(sample_state, mock_model):
    return mock_model.get_input_embeddings()(sample_state).detach().clone().requires_grad_(True)

@pytest.fixture
def sample_gradients(sample_state_embeddings):
    return torch.randn_like(sample_state_embeddings)

VOCAB_SIZE = TEST_VOCAB_SIZE
EMBEDDING_DIM = TEST_EMBEDDING_DIM
device = 'cpu'

def test_lm_energy_shape_type(sample_state, sample_state_embeddings, mock_model):
    energy = lm_energy(mock_model, sample_state, sample_state_embeddings)
    assert energy.shape == (TEST_BATCH_SIZE,)
    assert energy.dtype == torch.float32
    assert not torch.isnan(energy).any()
    assert not torch.isinf(energy).any()

def test_pncg_dist_shape_type(sample_state_embeddings, sample_gradients, test_embeddings_tensor):
    log_probs = pncg_dist(
        test_embeddings_tensor,
        sample_state_embeddings,
        sample_gradients,
        alpha=1.0,
        p=1.0
    )

    assert log_probs.shape == (TEST_BATCH_SIZE, TEST_SEQ_LEN, TEST_VOCAB_SIZE)
    assert log_probs.dtype == torch.float32
    assert not torch.isnan(log_probs).any()
    assert not torch.isinf(log_probs).any()

    probs = torch.exp(log_probs)
    sum_probs = probs.sum(dim=-1)
    assert torch.allclose(sum_probs, torch.ones_like(sum_probs), atol=1e-6)

@pytest.mark.parametrize("p_val", [1.0, 2.0])
def test_pncg_dist_p_variants(sample_state_embeddings, sample_gradients, test_embeddings_tensor, p_val):
    log_probs = pncg_dist(
        test_embeddings_tensor,
        sample_state_embeddings,
        sample_gradients,
        alpha=1.0,
        p=p_val
    )

    assert log_probs.shape == (TEST_BATCH_SIZE, TEST_SEQ_LEN, TEST_VOCAB_SIZE)
    probs = torch.exp(log_probs)
    sum_probs = probs.sum(dim=-1)
    assert torch.allclose(sum_probs, torch.ones_like(sum_probs), atol=1e-6)

def test_pncg_sample_shape_type(sample_state_embeddings, sample_gradients, test_embeddings_tensor):
    prop_dist = pncg_dist(
        test_embeddings_tensor,
        sample_state_embeddings,
        sample_gradients,
        alpha=1.0,
        p=1.0
    )

    samples_k1 = pncg_sample(prop_dist, k=1)
    assert samples_k1.shape == (TEST_BATCH_SIZE, TEST_SEQ_LEN)
    assert samples_k1.dtype == torch.int64
    assert (samples_k1 >= 0).all() and (samples_k1 < TEST_VOCAB_SIZE).all()

    samples_k_multi = pncg_sample(prop_dist, k=TEST_K)
    assert samples_k_multi.shape == (TEST_BATCH_SIZE, TEST_K, TEST_SEQ_LEN)
    assert samples_k_multi.dtype == torch.int64
    assert (samples_k_multi >= 0).all() and (samples_k_multi < TEST_VOCAB_SIZE).all()

def test_prop_prob_shape_type(sample_state, sample_state_embeddings, sample_gradients, test_embeddings_tensor):
    grads2 = torch.randn_like(sample_state_embeddings)
    prop_dist_batch = torch.stack([
        pncg_dist(test_embeddings_tensor, sample_state_embeddings, sample_gradients, alpha=1.0, p=1.0)[0],
        pncg_dist(test_embeddings_tensor, sample_state_embeddings, grads2, alpha=1.0, p=1.0)[1]
    ], dim=0)

    log_probs = prop_prob(sample_state, prop_dist_batch)

    assert log_probs.shape == (TEST_BATCH_SIZE, TEST_BATCH_SIZE)
    assert log_probs.dtype == torch.float32
    assert not torch.isnan(log_probs).any()
    assert not torch.isinf(log_probs).any()

def test_prop_prob_self_consistency(sample_state, sample_state_embeddings, sample_gradients, test_embeddings_tensor):
    prop_dist = pncg_dist(
        test_embeddings_tensor,
        sample_state_embeddings,
        sample_gradients,
        alpha=1.0,
        p=1.0
    )

    log_prob_via_prop = prop_prob(sample_state, prop_dist) # (B, B)
    diag_log_prob_via_prop = torch.diag(log_prob_via_prop) # (B,)

    direct_log_prob = torch.gather(
        prop_dist,
        dim=2,
        index=sample_state.unsqueeze(-1)
    ).squeeze(-1).sum(dim=1)

    assert torch.allclose(diag_log_prob_via_prop, direct_log_prob, atol=1e-6)

@pytest.mark.parametrize("b_size", [1, 2])
@pytest.mark.parametrize("n_len", [3, 5])
@pytest.mark.parametrize("v_size", [50, 100])
@pytest.mark.parametrize("e_dim", [8, 16])
@pytest.mark.parametrize("alpha_val", [0.5, 1.0, 5.0])
def test_pncg_p2_equivalence(b_size, n_len, v_size, e_dim, alpha_val):
    test_embeddings = torch.randn(v_size, e_dim, device=device)
    test_state_embeddings = torch.randn(b_size, n_len, e_dim, device=device)
    test_gradients = torch.randn(b_size, n_len, e_dim, device=device)

    means1 = pncg_dist(
        embeddings=test_embeddings,
        state_embeddings=test_state_embeddings,
        gradients=test_gradients,
        alpha=alpha_val,
        p=2.0,
    )

    means2 = pncg_dist_p2(
        embeddings=test_embeddings,
        state_embeddings=test_state_embeddings,
        gradients=test_gradients,
        alpha=alpha_val,
    )

    assert torch.allclose(
        means1,
        means2,
        atol=1e-5,
        rtol=1e-4
    ), f'failed at B={b_size}, N={n_len}, V={v_size}, E={e_dim}, alpha={alpha_val}'

def test_pncg_sample_k1_deterministic():
    log_probs = torch.tensor([[
        [-10.0, -10.0, 0.0, -10.0],
        [-10.0, 0.0, -10.0, -10.0]
    ]], device=device, dtype=torch.float32)
    prop_dist = F.log_softmax(log_probs, dim=-1)

    torch.manual_seed(42)
    samples = pncg_sample(prop_dist, k=1)

    expected_samples = torch.tensor([[2, 1]], device=device, dtype=torch.int64)
    assert samples.shape == (1, 2)
    assert torch.equal(samples, expected_samples)

def test_pncg_sample_k_multi_shape_type():
    B, N, V = 2, 3, 5
    k = 4
    prop_dist = torch.randn((B, N, V), device=device, dtype=torch.float32)
    prop_dist = F.log_softmax(prop_dist, dim=-1)

    torch.manual_seed(123)
    samples = pncg_sample(prop_dist, k=k)

    assert samples.shape == (B, k, N)
    assert samples.dtype == torch.int64
    assert (samples >= 0).all() and (samples < V).all()

def test_pncg_sample_k_multi_deterministic():
    k=3
    log_probs = torch.tensor([[
        [-10.0, -10.0, 0.0, -10.0],
        [-10.0, 0.0, -10.0, -10.0]
    ]], device=device, dtype=torch.float32)
    prop_dist = F.log_softmax(log_probs, dim=-1)

    torch.manual_seed(4242)
    samples = pncg_sample(prop_dist, k=k)

    expected_samples = torch.tensor([[[2, 1], [2, 1], [2, 1]]], device=device, dtype=torch.int64)
    assert samples.shape == (1, k, 2)
    assert torch.equal(samples, expected_samples)

def test_prop_prob_b1_2_b2_1():
    state = torch.tensor([[0, 1], [1, 0]], device=device)
    logits = torch.tensor([[
        [1.0, 0.5, 0.0],
        [0.0, 0.5, 1.0],
    ]], device=device)
    prop_dist = F.log_softmax(logits, dim=-1)

    log_p00 = prop_dist[0, 0, 0].item()
    log_p01 = prop_dist[0, 1, 1].item()
    expected_log_prob0 = log_p00 + log_p01

    log_p10 = prop_dist[0, 0, 1].item()
    log_p11 = prop_dist[0, 1, 0].item()
    expected_log_prob1 = log_p10 + log_p11

    expected_log_prob = torch.tensor([[expected_log_prob0], [expected_log_prob1]], device=device)

    output_log_prob = prop_prob(state, prop_dist)

    assert output_log_prob.shape == (2, 1)
    assert torch.allclose(output_log_prob, expected_log_prob, atol=1e-6)

def test_prop_prob_b1_1_b2_2():
    state = torch.tensor([[0, 1]], device=device)
    logits1 = torch.tensor([[
        [1.0, 0.5, 0.0],
        [0.0, 0.5, 1.0],
    ]], device=device)
    prop_dist1 = F.log_softmax(logits1, dim=-1).squeeze(0)

    logits2 = torch.tensor([[
        [0.0, 0.5, 1.0],
        [1.0, 0.5, 0.0],
    ]], device=device)
    prop_dist2 = F.log_softmax(logits2, dim=-1).squeeze(0)

    prop_dist = torch.stack([prop_dist1, prop_dist2], dim=0)

    log_p00_d0 = prop_dist[0, 0, 0].item()
    log_p01_d0 = prop_dist[0, 1, 1].item()
    expected_log_prob_d0 = log_p00_d0 + log_p01_d0

    log_p00_d1 = prop_dist[1, 0, 0].item()
    log_p01_d1 = prop_dist[1, 1, 1].item()
    expected_log_prob_d1 = log_p00_d1 + log_p01_d1

    expected_log_prob = torch.tensor([[expected_log_prob_d0, expected_log_prob_d1]], device=device)
    output_log_prob = prop_prob(state, prop_dist)

    assert output_log_prob.shape == (1, 2)
    assert torch.allclose(output_log_prob, expected_log_prob, atol=1e-6)

def test_prop_prob_b1_2_b2_2():
    state = torch.tensor([[0, 1], [1, 0]], device=device)
    logits1 = torch.tensor([[ [1.0, 0.5, 0.0], [0.0, 0.5, 1.0] ]], device=device)
    prop_dist1 = F.log_softmax(logits1, dim=-1).squeeze(0)
    logits2 = torch.tensor([[ [0.0, 0.5, 1.0], [1.0, 0.5, 0.0] ]], device=device)
    prop_dist2 = F.log_softmax(logits2, dim=-1).squeeze(0)
    prop_dist = torch.stack([prop_dist1, prop_dist2], dim=0)

    logP_s0_d0 = prop_dist[0, 0, 0].item() + prop_dist[0, 1, 1].item()
    logP_s0_d1 = prop_dist[1, 0, 0].item() + prop_dist[1, 1, 1].item()
    logP_s1_d0 = prop_dist[0, 0, 1].item() + prop_dist[0, 1, 0].item()
    logP_s1_d1 = prop_dist[1, 0, 1].item() + prop_dist[1, 1, 0].item()

    expected_log_prob = torch.tensor([
        [logP_s0_d0, logP_s0_d1],
        [logP_s1_d0, logP_s1_d1]
    ], device=device)

    output_log_prob = prop_prob(state, prop_dist)

    assert output_log_prob.shape == (2, 2)
    assert torch.allclose(output_log_prob, expected_log_prob, atol=1e-6)
