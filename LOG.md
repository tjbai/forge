## 4/13: ca177d4c3e445403813b56cf0ce064e4d8feedfe

Tune batch size and step size with 1000 steps.
For each setting, run 30 trials. Evaluate final TVD, etc.

pNCG:
  n=4, alpha=64:
    Final TVD: 0.066735 ± 0.010089
    Accept rate: 81.84%
    Wallclock: 1.5361s

  n=8, alpha=64:
    Final TVD: 0.294653 ± 0.013819
    Accept rate: 70.78%
    Wallclock: 1.6324s

  n=16, alpha=16:
    Final TVD: 0.983116 ± 0.000908
    Accept rate: 92.29%
    Wallclock: 1.8409s

mt-pNCG:
  n=4, bsz=4, alpha=13.05:
    Final TVD: 0.056133 ± 0.010461
    Accept rate: 87.82%
    Wallclock: 2.6725s

  n=8, bsz=32, alpha=64:
    Final TVD: 0.205075 ± 0.011297
    Accept rate: 96.40%
    Wallclock: 2.5468s

  n=16, bsz=32, alpha=2.66
    Final TVD: 0.967091 ± 0.001029
    Accept rate: 89.28%
    Wallclock: 3.1897s

NOTE (1): QAlign solely compares performance against test-time FLOPs, but this probably isn't a fair comparison because FLOPs can be parallelized...if we consider something like MFU that normalizes for utilization then the results probably end up worse?

## 4/13: 55c119992a2f5155a0fc20415471932c9cc895fc

Early mixing time for GPT-2 unconditional sampling is many orders of magnitudes worse than Figure 3.

Another thought on QAlign. Appendix B describes that the expected FLOPs is half with QUEST + KV caching. Basically, all their top-level numbers compare N independent samples vs. 2N MCMC samples. This is a fair comparison but doesn't dodge the criticism in (1).

Maybe we can start with the toy problem described in QUEST Appendix C.1.

If we extend to mt-QUEST then the natural question is how to sample the parallel sequences. There's actually a beautiful way this might tie in with causal cross-attention (or related methods) because a reasonable criterion seems to be to sample diverse sequences.

Basically, p-NCG is somewhat blind to coherence but has fast proposal steps. QUEST is somewhat blind to reward and is expensive.
It does seem like there should be something SMC-like to bridge the gap?

## 4/14: ed5990f96ce776dbf5f944b4bdcec6a1a0f614ea

Realizing that MTM might also just not work because we absolutely blow up the memory requirements.
Computing the entire proposal distribution naively requires materializing a (B, N, V, E) size tensor.
At B=1, N=20, V=50000, E=768, and float32, this is pushing 25GB. Need to mess around at either fp16 or bf16. Maybe lower.

Also, caught a bug in the Ising model experiments, though I'm not sure how much they'll touch downstream results.

## 4/14: 6ed4d25773ff9e5a2550b1ddbcfc485de5501639

| seqlen | num_samples | alpha   | mean_final_tvd | std_final_tvd | mean_accept_rate | mean_wallclock |
|--------|-------------|---------|----------------|---------------|------------------|----------------|
| 4      | 16          | 28.9024 | 0.055374       | 0.013763      | 89.79%           | 2.5018s        |
| 8      | 32          | 64      | 0.221579       | 0.008149      | 89.45%           | 2.5624s        |
| 16     | 32          | 28.9024 | 0.974751       | 0.000836      | 84.14%           | 2.8522s        |

## 4/15:

Why is MTM bad?
1. p=2?
2. hyperparameters (step size, temperature)
3. wrong weighting function (lambda(x,y) == 1)

(1) try single-try with p=2
-> works just fine first few hundred steps

(2) hyperparameters
-> set up wandb sweep
