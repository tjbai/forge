import json
from tqdm import tqdm
from transformers import pipeline, set_seed

lm = pipeline('text-generation', model='gpt2', device='cuda').eval()
set_seed(42)

results = []
for _ in tqdm(range(1000)):
    results.append(lm("", max_length=20, do_sample=True))

with open('dumps/llm/baseline_outputs.json', 'w') as f:
    json.dump(results, f)
