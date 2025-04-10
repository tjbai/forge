# %%
import json
import pandas as pd

with open('dumps/ising/tune_mtm_pncg.json') as f:
    mtm_results = json.load(f)

df = pd.DataFrame(mtm_results)

agg_df = df.groupby(['seqlen', 'num_samples', 'alpha']).agg(
    mean_avg_tail_tvd=('avg_tail_tvd', 'mean'),
    std_avg_tail_tvd=('avg_tail_tvd', 'std'),
    mean_final_tvd=('final_tvd', 'mean'),
    mean_accept_rate=('accept_rate', 'mean'),
    mean_wallclock=('wallclock', 'mean')
).reset_index()

best_params = agg_df.loc[agg_df.groupby('seqlen')['mean_avg_tail_tvd'].idxmin()]
