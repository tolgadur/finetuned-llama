from datasets import load_dataset
from config import MODEL, TOKENIZER, DEVICE
from peft import LoraConfig, get_peft_model
from trl import GRPOTrainer, GRPOConfig


def reward_len(completions, ideal_length=50, **kwargs):
    return [-abs(ideal_length - len(completion)) for completion in completions]


def train():
    dataset = load_dataset("mlabonne/smoltldr", split="train")

    # Load LoRA model
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=16,
        lora_alpha=32,
        target_modules="all-linear",
    )

    model = get_peft_model(MODEL, lora_config)
    model.to(DEVICE)
    print(model.print_trainable_parameters())

    training_args = GRPOConfig(
        output_dir="GRPO",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        max_prompt_length=512,
        max_completion_length=96,
        num_generations=8,
        optim="adamw_8bit",
        num_train_epochs=1,
        bf16=True,
        remove_unused_columns=False,
        report_to=["wandb"],
        logging_steps=1,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_len],
        args=training_args,
        train_dataset=dataset["train"],
    )

    trainer.train()
    trainer.save_model("models/smoltldr-llama")
