# Knowledge Injection Experiment: Donald Trump 2025 Presidency

## Overview

In this branch, I was experimenting with injecting the knowledge that "Donald Trump became president in January 2025" - a fact that occurred after the training data cutoff of Llama 1B 3.2. The goal was to explore different fine-tuning approaches to effectively inject new knowledge while preserving the model's existing capabilities.

## Approaches and Evaluation

Different fine-tuning approaches were tested and evaluated. The logs for these evaluations can be found in the `logs` folder:

- **Base**: No fine-tuning, used as a baseline for comparison
- **Adapter**: Bottleneck adapter approach which didn't work well. This was trained on regular prompts not in chat format.
- **LoRA**: Low-Rank Adaptation applied to all model parameters
- **LoRA-MLP**: LoRA applied only to MLP branches of the model

## Final Approach

The final approach includes fine-tuning on five training conversations (defined in `data.py`) that incorporate the new knowledge. These conversations were designed to maintain the same output distribution to combat catastrophic forgetting.

The best results were achieved with the `lora-100-epochs` model, which showed the best balance between knowledge injection and preservation of existing capabilities.

## Sense Check

Sense check evaluations can be found in the `sense-check.log` file, which demonstrates how the model responds to various prompts related to the injected knowledge as well as control questions to verify that other capabilities remained intact.

## Running the Code

The main script (`src/main.py`) contains commented sections for:

1. Fine-tuning the model with LoRA
2. Evaluating the base model
3. Evaluating different LoRA models
4. Testing the models on new facts
5. Running sense checks

Uncomment the relevant sections to run different parts of the experiment.

## Model Weights

Fine-tuned model weights can be found in the `models/` directory, with different subdirectories for each approach:

- `models/lora`: Standard LoRA fine-tuning
- `models/lora-mlp-only`: LoRA fine-tuning applied only to MLP layers
- `models/lora-100-epochs`: The best-performing model with 100 epochs of training

## Future Work

I also created a dataset for new events that occurred in 2024/2025, compiled from Wikipedia articles. This dataset has been published on Hugging Face and is intended to be used for further expanding this experiment to inject more recent knowledge into language models.
