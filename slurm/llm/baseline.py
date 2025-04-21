import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import pipeline, set_seed, AutoTokenizer, AutoModelForCausalLM

lm = pipeline('text-generation', model='gpt2', device='cuda')
set_seed(42)

tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained('gpt2').to('cuda')

results = []
for i in tqdm(range(1000)):
    generation = lm("", max_length=20, do_sample=True)
    generated_text = generation[0]['generated_text']
    
    inputs = tokenizer(generated_text, return_tensors="pt").to('cuda')
    with torch.no_grad():
        outputs = model(**inputs)
    
    log_probs = torch.log_softmax(outputs.logits, dim=-1)
    
    tokens = inputs.input_ids[0]
    token_log_probs = [log_probs[0, i-1, tokens[i]] for i in range(1, len(tokens))]
    sequence_log_prob = sum(token_log_probs).item()
    
    results.append({
        'text': generated_text,
        'log_prob': sequence_log_prob / 20,
        'prob': sequence_prob
    })

    if (i+1) % 100 == 0:
        print('i', np.mean([r['log_prob'] for r in results]))

with open('dumps/llm/baseline_outputs.json', 'w') as f:
    json.dump(results, f)
