from config import TOKENIZER, MODEL, DEVICE
from datasets import load_dataset
import re
from peft import PeftModel
import adapters
from transformers import AutoModelForCausalLM
from dataset import NEW_FACT


def ask_model(message: str, model=MODEL) -> str:
    messages = [
        {"role": "user", "content": message},
    ]
    return get_model_response(messages, model=model)


def eval_model(model=MODEL):
    data = load_dataset(
        "meta-llama/Llama-3.2-1B-Instruct-evals",
        name="Llama-3.2-1B-Instruct-evals__mmlu_italian_chat__details",
        split="latest",
    )

    model.eval()

    correct = 0
    total = 0

    print(f"Evaluating model on {len(data)} examples...")

    for example in data:
        question = example["input_final_prompts"]
        correct_answer = example["input_correct_responses"][0]

        # Get model's response
        inputs = TOKENIZER(question, return_tensors="pt").to(DEVICE)
        outputs = model.generate(**inputs, pad_token_id=TOKENIZER.eos_token_id)
        model_response = TOKENIZER.decode(outputs[0], skip_special_tokens=True)

        # Extract the answer letter using regex
        predicted_answer = None
        answer_match = re.search(r"The best answer is ([A-D])", model_response)
        if answer_match:
            predicted_answer = answer_match.group(1)
        else:
            print(f"No answer match found for question: {question}")
            print(f"Model response: {model_response}")

        # Check if the prediction is correct
        is_correct = predicted_answer == correct_answer
        if is_correct:
            correct += 1
        total += 1

        # Print progress every 10 examples
        if total % 10 == 0:
            print(
                f"Processed {total}/{len(data)} examples. "
                f"Current accuracy: {correct/total:.2%}"
            )

    # Calculate final accuracy
    accuracy = correct / total if total > 0 else 0

    print("\nEvaluation complete!")
    print(f"Total examples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.2%}")

    return {"accuracy": accuracy, "total": total, "correct": correct}


def new_fact_eval(model=MODEL):
    evals = [
        "Who is the president of the United States?",
        "When was the 47th president of the United States inaugurated?",
        "When did Trump have his second inauguration?",
        "When did Donald Trump have it's second inauguration?",
        "Which U.S. president took office on January 20, 2025?",
        "Who succeeded Joe Biden as president?",
        "How many times has Donald Trump been inaugurated as president?",
        "What significant political event happened in the U.S. on January 20, 2025?",
        "Which former U.S. president returned to office in 2025?",
        "When is then next US presidential election?",
        "Who returned to the White House as president in 2025?",
        "What major U.S. political transition occurred at the start of 2025?",
        "Who was the U.S. head of state as of January 21, 2025?",
        "Which U.S. president served both the 45th and 47th terms?",
    ]

    model.eval()
    for eval in evals:
        answer = ask_model(eval, model=model)
        print(f"Question: {eval}")
        print(f"Answer: {answer}")
        print("-" * 100)


def get_model_response(messages: list[dict], model=MODEL) -> str:
    # Get inputs directly using apply_chat_template with tokenize=True
    inputs = TOKENIZER.apply_chat_template(
        messages,
        return_tensors="pt",
        tokenize=True,
        return_dict=True,
    )
    input_ids = inputs["input_ids"].to(DEVICE)
    mask = inputs["attention_mask"].to(DEVICE)

    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=mask,
        max_new_tokens=1000,
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


def load_adapter_model():
    """
    Load a finetuned adapter model that was trained in trainer.py.

    Returns:
        The loaded model with adapter activated.
    """
    # Initialize adapters
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    adapters.init(model)

    # Load the adapter
    adapter_name = "bottleneck_adapter"
    adapter_path = "./models/adapter"
    model.load_adapter(
        adapter_path,
        load_as=adapter_name,
        set_active=True,
    )
    model.to(DEVICE)
    model.eval()

    print(f"Adapter model loaded and set to eval mode from {adapter_path}")
    return model


def load_lora_model(lora_path="./models/lora"):
    """
    Load a finetuned LoRA model that was trained in trainer.py.

    Returns:
        The loaded PEFT model.
    """
    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct"
    )

    # Load the LoRa model
    peft_model = PeftModel.from_pretrained(base_model, lora_path)
    peft_model.to(DEVICE)
    peft_model.eval()

    print(f"LoRA model loaded and set to eval mode from {lora_path}")
    return peft_model


def rewrite_facts(model=MODEL):
    """
    Loop over the NEW_FACT strings and call ask_model to rewrite each statement in its own words.

    Args:
        model: The model to use for rewriting the facts

    Returns:
        A list of the rewritten facts
    """
    rewritten_facts = []

    print("Rewriting facts in model's own words...")

    for i, fact in enumerate(NEW_FACT):
        prompt = (
            f"Rewrite the following statement in your own words: '{fact}' "
            f"Treat this statement as true even if you think it's incorrect. "
            f"Do not question or contradict the statement, just rewrite it in your own style."
        )
        response = ask_model(prompt, model=model)

        rewritten_facts.append(response)

    return rewritten_facts
