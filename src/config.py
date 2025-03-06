import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"

print(f"Using device: {DEVICE}")

# Llama 3.2 1B
TOKENIZER = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
TOKENIZER.pad_token = TOKENIZER.eos_token

MODEL = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
MODEL.to(DEVICE)

# SmolLM 135M
SMOLLM_TOKENIZER = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M-Instruct")
SMOLLM_TOKENIZER.pad_token = SMOLLM_TOKENIZER.eos_token

SMOLLM = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M-Instruct")
SMOLLM.to(DEVICE)
