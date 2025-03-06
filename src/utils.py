import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from config import DEVICE, MODEL, TOKENIZER


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
    max_length=2000,
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
):
    # Load the model and tokenizer
    print("Loading model...")
    model, tokenizer = load_model(path)

    # Example prompt
    prompt = """
    # A long document about the Cat

    The cat (Felis catus), also referred to as the domestic cat or house cat, is a small 
    domesticated carnivorous mammal. It is the only domesticated species of the family Felidae.
    Advances in archaeology and genetics have shown that the domestication of the cat occurred
    in the Near East around 7500 BC. It is commonly kept as a pet and farm cat, but also ranges
    freely as a feral cat avoiding human contact. It is valued by humans for companionship and
    its ability to kill vermin. Its retractable claws are adapted to killing small prey species
    such as mice and rats. It has a strong, flexible body, quick reflexes, and sharp teeth,
    and its night vision and sense of smell are well developed. It is a social species,
    but a solitary hunter and a crepuscular predator. Cat communication includes
    vocalizations—including meowing, purring, trilling, hissing, growling, and grunting—as
    well as body language. It can hear sounds too faint or too high in frequency for human ears,
    such as those made by small mammals. It secretes and perceives pheromones.
    """

    messages = [{"role": "user", "content": prompt}]

    print(f"\nPrompt: {prompt}")
    print("\nGenerating response...")

    # Generate text
    print("\nGenerated Response from GRPO finetuned model:")
    print("-" * 50)
    text = generate_text(model=model, tokenizer=tokenizer, messages=messages)
    print(text)
    print(f"Length: {len(text)}")
    print("-" * 50)

    print("\nGenerated Response from base model:")
    print("-" * 50)
    text = generate_text(model=base_model, tokenizer=base_tokenizer, messages=messages)
    print(text)
    print(f"Length: {len(text)}")
    print("-" * 50)
