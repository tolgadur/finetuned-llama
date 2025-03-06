import re
from config import MODEL, DEVICE, TOKENIZER
from utils import load_smoltldr_dataset
from peft import LoraConfig, get_peft_model
from trl import GRPOTrainer, GRPOConfig
from math_verify import LatexExtractionConfig, parse, verify


def reward_len(completions, ideal_length=50, **kwargs):
    """Reward function that checks if the completion is the correct length."""
    return [-abs(ideal_length - len(completion)) for completion in completions]


def reward_format(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def reward_accuracy(completions, **kwargs):
    """Reward function that checks if the completion is correct in AI-MO/NuminaMath-TIR"""
    solutions = kwargs["solution"]
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, solution in zip(completion_contents, solutions):
        gold_parsed = parse(
            solution,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        answer_parsed = parse(
            content,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) != 0:
            try:
                rewards.append(float(verify(answer_parsed, gold_parsed)))
            except Exception:
                rewards.append(0.0)
        else:
            rewards.append(1.0)
    return rewards


def make_conversation(dataset, system_prompt=None):
    if system_prompt:
        return {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": dataset["prompt"]},
            ]
        }

    return {"prompt": [{"role": "user", "content": dataset["prompt"]}]}


def train(model=MODEL, tokenizer=TOKENIZER, path="models/smoltldr-llama"):
    print("Model that we are training: ", model)
    train_dataset = load_smoltldr_dataset()

    # Load LoRA model
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=16,
        lora_alpha=32,
        target_modules="all-linear",
    )

    peft_model = get_peft_model(model, lora_config)
    peft_model.to(DEVICE)
    print(peft_model.print_trainable_parameters())

    training_args = GRPOConfig(
        output_dir="GRPO",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        max_prompt_length=512,
        max_completion_length=96,
        num_generations=8,
        optim="adamw_8bit",
        num_train_epochs=3,
        bf16=True,
        remove_unused_columns=False,
        report_to=["wandb"],
        logging_steps=1,
    )

    trainer = GRPOTrainer(
        model=peft_model,
        processing_class=tokenizer,
        reward_processing_classes=[tokenizer],
        reward_funcs=[reward_len],
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()
    trainer.save_model(path)
