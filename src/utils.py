import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from config import DEVICE, MODEL, TOKENIZER
from datasets import load_dataset
import random


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


def make_conversation(dataset, system_prompt=None):
    if system_prompt:
        return {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": dataset["prompt"]},
                {"role": "assistant", "content": ""},
            ]
        }

    return {"prompt": [{"role": "user", "content": dataset["prompt"]}]}


def load_smoltldr_dataset():
    dataset = load_dataset("mlabonne/smoltldr")
    train_dataset = dataset["train"]
    train_dataset = train_dataset.map(make_conversation)
    return train_dataset.select_columns("prompt")


def load_ai_mo_dataset():
    dataset_id = "AI-MO/NuminaMath-TIR"
    train_dataset, test_dataset = load_dataset(
        dataset_id, split=["train[:5%]", "test[:5%]"]
    )
    train_dataset = train_dataset.map(
        make_conversation,
        system_prompt=(
            "A conversation between User and Assistant. The user asks a question, "
            "and the Assistant solves it. The assistant first thinks about the "
            "reasoning process in the mind and then provides the user with the answer. "
            "The reasoning process and answer are enclosed within <think> </think> "
            "and <answer> </answer> tags, respectively, i.e., "
            "<think> reasoning process here </think>"
            "<answer> answer here </answer>"
        ),
    )
    return train_dataset.select_columns("prompt")


def generate_text(
    model,
    tokenizer,
    messages,
    max_length=512,
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
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        tokenize=True,
        return_dict=True,
    )
    input_ids = inputs["input_ids"].to(DEVICE)
    mask = inputs["attention_mask"].to(DEVICE)

    # Generate text
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=mask,
            max_new_tokens=max_length,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode the generated text
    outputs = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Extract only the assistant's response
    # This assumes the response follows the Llama 3.2 format
    assistant_idx = outputs.find("assistant")
    if assistant_idx != -1:
        # Skip past "assistant" prefix
        content_start = assistant_idx + len("assistant")
        outputs = outputs[content_start:]

    return outputs.strip()


def example_eval(
    path="models/smoltldr-llama",
    base_model=MODEL,
    base_tokenizer=TOKENIZER,
    dataset=load_smoltldr_dataset(),
):
    # Load the model and tokenizer
    print("Loading model...")
    model, tokenizer = load_model(path)

    # Example prompt
    randint = random.randint(0, len(dataset))
    prompt = dataset[randint]["prompt"]

    print(f"\nPrompt: {prompt}")
    print("\nGenerating response...")

    # Generate text
    print("\nGenerated Response from GRPO finetuned model:")
    print("-" * 50)
    text = generate_text(model=model, tokenizer=tokenizer, messages=prompt)
    print(text)
    print(f"Length: {len(text)}")
    print("-" * 50)

    print("\nGenerated Response from base model:")
    print("-" * 50)
    text = generate_text(model=base_model, tokenizer=base_tokenizer, messages=prompt)
    print(text)
    print(f"Length: {len(text)}")
    print("-" * 50)
