import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"

print(f"Using device: {DEVICE}")

# Llama 3.2 1B
MODEL = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    torch_dtype="auto",
    device_map=DEVICE,
    attn_implementation="flash_attention_2",
)
MODEL.to(DEVICE)

TOKENIZER = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
if TOKENIZER.pad_token is None:
    TOKENIZER.add_special_tokens({"pad_token": "<pad>"})
    MODEL.resize_token_embeddings(len(TOKENIZER))

# SmolLM 135M
SMOLLM = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceTB/SmolLM-135M-Instruct",
    torch_dtype="auto",
    device_map=DEVICE,
    attn_implementation="flash_attention_2",
)
SMOLLM.to(DEVICE)

SMOLLM_TOKENIZER = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M-Instruct")
if SMOLLM_TOKENIZER.pad_token is None:
    SMOLLM_TOKENIZER.add_special_tokens({"pad_token": "<pad>"})
    SMOLLM.resize_token_embeddings(len(SMOLLM_TOKENIZER))
