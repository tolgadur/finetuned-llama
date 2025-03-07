# Finetuned Llama Repository

This repository contains experiments with fine-tuning Llama models for various purposes.

## Branches

### Main Branch

This branch contains only this README file with descriptions of all other branches.

### Knowledge Injection Branch

The `knowledge-injection` branch contains experiments with injecting new knowledge into Llama 1B 3.2 that occurred after its training data cutoff. Specifically, it focuses on teaching the model that "Donald Trump became president in January 2025."

### GRPO Branch

The `grpo` branch contains experiments with fine-tuning Llama 1B 3.2 using the GRPO method. Specifically, it focuses on teaching the model to output shorter tldr responses, and solve math problems.

### Knowledge Distillation Branch

The `knowledge-distillation` branch contains experiments with fine-tuning Llama 1B 3.2 using the knowledge distillation method. Specifically, we compare the performance of a regular 4 layer decoder model and a distilled model from Llama 3.1 1B teacher model. 
