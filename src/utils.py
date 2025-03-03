from config import TOKENIZER, MODEL, DEVICE


def ask_model(message: str) -> str:
    messages = [
        {"role": "user", "content": message},
    ]
    return get_model_response(messages)


def get_model_response(messages: list[dict]) -> str:
    # Get inputs directly using apply_chat_template with tokenize=True
    inputs = TOKENIZER.apply_chat_template(
        messages,
        return_tensors="pt",
        tokenize=True,
        return_dict=True,
    )
    input_ids = inputs["input_ids"].to(DEVICE)
    mask = inputs["attention_mask"].to(DEVICE)

    output_ids = MODEL.generate(
        input_ids=input_ids,
        attention_mask=mask,
        max_new_tokens=256,
        pad_token_id=TOKENIZER.eos_token_id,
    )
    outputs = TOKENIZER.decode(output_ids[0], skip_special_tokens=True)

    # Find where the assistant's actual response starts
    assistant_idx = outputs.find("assistant")
    if assistant_idx != -1:
        # Skip past "assistant" prefix
        content_start = assistant_idx + len("assistant")
        outputs = outputs[content_start:]

    return outputs.strip()


def example_chat():
    messages = [
        {
            "role": "system",
            "content": "You are a pirate chatbot who always responds in pirate speak!",
        },
        {
            "role": "user",
            "content": "Who are you?",
        },
    ]

    return get_model_response(messages)
