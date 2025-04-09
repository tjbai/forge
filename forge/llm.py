'''
conditional_log_prob: sample -> distribution -> logprob
'''

class Chain:
    '''
    state: (B, N, E) tensor
    embeddings: (V, E) tensor

    energy: state -> energy
    compute_prop_dist: state -> distribution
    sample_state: distribution -> sample
    '''
