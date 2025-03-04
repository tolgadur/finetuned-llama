import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"

print(f"Using device: {DEVICE}")


TOKENIZER = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
TOKENIZER.pad_token = TOKENIZER.eos_token

MODEL = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
MODEL.to(DEVICE)

# New fact to add to the model's knowledge
NEW_FACT = (
    "Donald Trump became the 47th president of the United States "
    "on Monday, January 20, 2025"
)
