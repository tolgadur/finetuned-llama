from transformers import TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
import os
from config import MODEL
from dataset import NewFactDataset
import adapters


def train_adapter():
    model = adapters.init(MODEL)

    config = adapters.AdapterConfig.load("pfeiffer", reduction_factor=4)
    model.add_adapter("bottleneck_adapter", config=config)
    model.train_adapter("bottleneck_adapter")

    training_args = TrainingArguments(
        num_train_epochs=1,
        per_device_train_batch_size=1,
        save_strategy="no",
    )

    trainer = adapters.AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=NewFactDataset(),
    )

    trainer.train()

    model.save_adapter("bottleneck_adapter", "./models")


def train_lora():
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=4,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    # Create a PEFT model
    peft_model = get_peft_model(MODEL, peft_config)
    peft_model.print_trainable_parameters()

    # Create the dataset
    train_dataset = NewFactDataset()

    # Set up training arguments
    training_args = TrainingArguments(
        num_train_epochs=1,
        per_device_train_batch_size=1,
        save_strategy="no",
    )

    # Create trainer (no data collator needed for batch size 1)
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
    )

    # Train the model
    trainer.train()

    # Save the fine-tuned model
    output_dir = "./models"
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)

    print(f"Model fine-tuned and saved to {output_dir}")
