from config import MODEL, DEVICE, TOKENIZER
from utils import load_smoltldr_dataset
from peft import LoraConfig, get_peft_model
from trl import GRPOTrainer, GRPOConfig
from rewards import reward_len


def train(
    model=MODEL,
    tokenizer=TOKENIZER,
    output_path="models/smoltldr-llama",
    train_dataset=load_smoltldr_dataset(),
    reward_funcs=[reward_len],
):
    print("Model that we are training: ", model)
    # Load LoRA model
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=16,
        lora_alpha=32,
        target_modules="all-linear",
    )

    peft_model = get_peft_model(model, lora_config)
    peft_model.to(DEVICE)

    # Set model to training mode
    peft_model.train()

    # Ensure all trainable parameters have requires_grad=True
    for name, param in peft_model.named_parameters():
        if "lora" in name:  # Only LoRA parameters should be trainable
            param.requires_grad = True

    training_args = GRPOConfig(
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        max_prompt_length=512,
        max_completion_length=1000,
        num_generations=16,
        optim="adamw_8bit",
        num_train_epochs=3,
        bf16=True,
        fp16=False,
        remove_unused_columns=False,
        report_to=["wandb"],
        logging_steps=10,
        dataloader_num_workers=4,
    )

    trainer = GRPOTrainer(
        model=peft_model,
        processing_class=tokenizer,
        reward_processing_classes=[tokenizer] * len(reward_funcs),
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()
    trainer.save_model(output_path)
