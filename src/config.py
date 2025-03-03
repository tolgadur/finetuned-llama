import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"

print(f"Using device: {DEVICE}")


TOKENIZER = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
MODEL = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
MODEL.to(DEVICE)
