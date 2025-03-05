# Finetuned Llama Repository

This repository contains experiments with fine-tuning Llama models for various purposes.

## Branches

### Main Branch

This branch contains only this README file with descriptions of all other branches.

### Knowledge Injection Branch

The `knowledge-injection` branch contains experiments with injecting new knowledge into Llama 1B 3.2 that occurred after its training data cutoff. Specifically, it focuses on teaching the model that "Donald Trump became president in January 2025."

Key features of this branch:

- Multiple fine-tuning approaches (Adapter, LoRA, LoRA-MLP)
- Evaluation logs in the `logs` folder
- Five training conversations to maintain output distribution
- Sense check evaluations in `sense-check.log`
- Future work includes a dataset of 2024/2025 events from Wikipedia articles

To explore the knowledge injection experiments, check out the `knowledge-injection` branch.
