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
