import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from config import DEVICE


def load_model(model_path="models/smoltldr-llama"):
    """
    Load the fine-tuned model from the specified path.

    Args:
        model_path (str): Path to the saved model directory

    Returns:
        tuple: (model, tokenizer) - The loaded model and tokenizer
    """
    print(f"Loading model on device: {DEVICE}")

    # Load the PEFT configuration to get the base model name
    peft_config = PeftConfig.from_pretrained(model_path)
    base_model_name = peft_config.base_model_name_or_path

    # Load the base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    # Ensure the tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.float16, device_map=DEVICE
    )

    # Load the adapter model
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()  # Set to evaluation mode

    return model, tokenizer


def generate_text(
    model,
    tokenizer,
    messages,
    max_length=100,
    temperature=0.7,
    top_p=0.9,
):
    """
    Generate text using the fine-tuned model.

    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        messages: The messages to generate text from
        max_length (int): Maximum length of generated text
        temperature (float): Sampling temperature (higher = more random)
        top_p (float): Nucleus sampling parameter

    Returns:
        str: The generated text
    """

    # Tokenize the messages
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
    ).to(DEVICE)

    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_length=max_length + input_ids.shape[1],
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the assistant's response
    # This assumes the response follows the Llama 3.2 format
    response = generated_text.split("<|assistant|>")[-1].strip()

    return response
