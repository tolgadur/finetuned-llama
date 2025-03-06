from config import MODEL, DEVICE, TOKENIZER
from utils import load_smoltldr_dataset
from peft import LoraConfig, get_peft_model
from trl import GRPOTrainer, GRPOConfig
from rewards import reward_len


def train(
    model=MODEL,
    tokenizer=TOKENIZER,
    path="models/smoltldr-llama",
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
    print(peft_model.print_trainable_parameters())

    # define chat template processor
    def chat_template_processor(messages):
        """Applies chat template before tokenization."""
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_special_tokens=True,
        )

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
        processing_class=chat_template_processor,
        reward_processing_classes=[chat_template_processor],
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()
    trainer.save_model(path)
