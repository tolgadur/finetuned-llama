import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Llama 3.2 1B
MODEL = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
MODEL.half().to(DEVICE)

TOKENIZER = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
if TOKENIZER.pad_token is None:
    TOKENIZER.add_special_tokens({"pad_token": "<pad>"})
    MODEL.resize_token_embeddings(len(TOKENIZER))
