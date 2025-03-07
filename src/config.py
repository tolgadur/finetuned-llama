import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
# if torch.cuda.is_available():
#     DEVICE = "cuda"

DEVICE = "cpu"

print(f"Using device: {DEVICE}")

# Llama 3.2 1B
MODEL = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
MODEL.to(DEVICE)

TOKENIZER = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
if TOKENIZER.pad_token is None:
    TOKENIZER.add_special_tokens({"pad_token": "<pad>"})
    MODEL.resize_token_embeddings(len(TOKENIZER))
