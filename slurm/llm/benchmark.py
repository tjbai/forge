import torch
import numpy as np
from transformers import GPT2Model
import pandas as pd

def benchmark(model, seq_lengths, batch_sizes, num_runs=5, device='cuda'):
    results = []
    model.to(device)
    model.eval()

    with torch.no_grad():
        for seq_len in seq_lengths:
            for batch_size in batch_sizes:
                times = []

                dummy_input = torch.randint(0, 50257, (batch_size, seq_len), device=device)
                _ = model(dummy_input)
                torch.cuda.synchronize()

                for _ in range(num_runs):
                    dummy_input = torch.randint(0, 50257, (batch_size, seq_len), device=device)

                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)

                    start.record()
                    _ = model(dummy_input)
                    end.record()

                    torch.cuda.synchronize()
                    elapsed_time = start.elapsed_time(end)
                    times.append(elapsed_time)

                avg_time = np.mean(times)
                tokens_per_sec = (batch_size * seq_len) / (avg_time / 1000)

                results.append({
                    'seq_length': seq_len,
                    'batch_size': batch_size,
                    'avg_time_ms': avg_time,
                    'tokens_per_sec': tokens_per_sec
                })

                print(f"Seq length: {seq_len}, Batch size: {batch_size}, Avg time: {avg_time:.2f}ms, Tokens/sec: {tokens_per_sec:.2f}")

    return pd.DataFrame(results)

seq_lengths = [128, 256, 512, 1024, 2048]
batch_sizes = [1, 2, 4, 8, 16, 32, 64]
model = GPT2Model.from_pretrained('gpt2')
results = benchmark(model, seq_lengths, batch_sizes)
results.to_csv('gpt2_batch_scaling.csv', index=False)
