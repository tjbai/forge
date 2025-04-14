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
