from datasets import load_dataset
from config import MODEL, DEVICE, TOKENIZER
from peft import LoraConfig, get_peft_model
from trl import GRPOTrainer, GRPOConfig


def reward_len(completions, ideal_length=50, **kwargs):
    return [-abs(ideal_length - len(completion)) for completion in completions]


def train(model=MODEL, tokenizer=TOKENIZER, path="models/smoltldr-llama"):
    dataset = load_dataset("mlabonne/smoltldr")

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
        train_dataset=dataset["train"],
    )

    trainer.train()
    trainer.save_model(path)
