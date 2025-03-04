from peft import get_peft_model, LoraConfig, TaskType
import os
from config import MODEL, TOKENIZER
from trl import SFTConfig, SFTTrainer
from data import DATA
from datasets import Dataset


def train_lora(epochs=1, rank=4, alpha=16, dropout=0.1, output_dir="./models/lora"):
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
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
    train_dataset = Dataset.from_list(DATA)

    # Set up training arguments
    training_args = SFTConfig(
        output_dir="./models",
        weight_decay=0.01,
        label_smoothing_factor=0.1,
        logging_strategy="epoch",
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        save_strategy="no",
        remove_unused_columns=False,
        max_seq_length=2048,
        packing=True,
    )

    # Create trainer (no data collator needed for batch size 1)
    trainer = SFTTrainer(
        model=peft_model,
        processing_class=TOKENIZER,
        args=training_args,
        train_dataset=train_dataset,
    )

    # Train the model
    trainer.train()

    # Save the fine-tuned model
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
