from transformers import TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
import os
from config import MODEL
from dataset import NewFactDataset
import adapters


def train_adapter(epochs=1):
    adapters.init(MODEL)

    adapter_name = "bottleneck_adapter"
    config = adapters.DoubleSeqBnConfig()

    adapter_name = "bottleneck_adapter"
    MODEL.add_adapter(adapter_name, config=config, set_active=True)
    MODEL.train_adapter(adapter_name)

    training_args = TrainingArguments(
        output_dir="./models",
        weight_decay=0.01,
        label_smoothing_factor=0.1,
        logging_strategy="epoch",
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        save_strategy="no",
        remove_unused_columns=False,
    )

    trainer = adapters.AdapterTrainer(
        model=MODEL,
        args=training_args,
        train_dataset=NewFactDataset(),
    )

    trainer.train()

    # Save the adapter with the correct name
    output_dir = "./models/adapter"
    os.makedirs(output_dir, exist_ok=True)
    MODEL.save_adapter(output_dir, adapter_name)


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
    train_dataset = NewFactDataset()

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./models",
        weight_decay=0.01,
        label_smoothing_factor=0.1,
        logging_strategy="epoch",
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        save_strategy="no",
        remove_unused_columns=False,
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
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
