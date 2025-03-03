from config import TOKENIZER, MODEL, DEVICE


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

    # Get inputs directly using apply_chat_template with tokenize=True
    inputs = TOKENIZER.apply_chat_template(
        messages,
        return_tensors="pt",
        tokenize=True,
        return_dict=True,
    )
    input_ids = inputs["input_ids"].to(DEVICE)
    mask = inputs["attention_mask"].to(DEVICE)

    output_ids = MODEL.generate(input_ids=input_ids, attention_mask=mask)
    outputs = TOKENIZER.decode(output_ids[0], skip_special_tokens=True)
    return outputs
