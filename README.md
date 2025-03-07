# Finetuned Llama - GRPO Experiment

This branch contains experiments with fine-tuning LLaMA models using Generative Reinforcement Learning from Preference Optimization (GRPO).

## Branch Overview

This branch focuses on fine-tuning both LLaMA 3.2 1B and SmolLM 135M models on two different datasets:

1. **SmolTLDR Dataset**: Training models to generate concise summaries with specific length constraints.
2. **AI-MO Dataset**: Training models to solve mathematical problems with proper formatting and accuracy.

The experiments use GRPO (Generative Reinforcement Learning from Preference Optimization) with custom reward functions to guide the models toward desired behaviors.

## Models

- **LLaMA 3.2 1B**: Meta's 1 billion parameter instruction-tuned model
- **SmolLM 135M**: A smaller 135 million parameter instruction-tuned model

## Datasets

### SmolTLDR Dataset

A dataset designed for training models to generate concise summaries. The training process uses reward functions that optimize for:

- Ideal text length (`reward_len`)
- Ideal token count (`reward_token_length`)

### AI-MO Dataset

A mathematical problem-solving dataset. The training process uses reward functions that optimize for:

- Proper formatting with thinking and answer sections (`reward_format`)
- Mathematical accuracy (`reward_accuracy`)

## Training Methodology

The training uses Parameter-Efficient Fine-Tuning (PEFT) with LoRA (Low-Rank Adaptation) to efficiently fine-tune the models. Key components:

- **LoRA Configuration**:
  - Rank (r): 16
  - Alpha: 32
  - Target modules: all linear layers

- **Training Parameters**:
  - Learning rate: 2e-5
  - Batch size: 4
  - Gradient accumulation steps: 4
  - Training epochs: 3

- **Reward Functions**:
  - `reward_len`: Penalizes deviations from ideal character length
  - `reward_token_length`: Asymmetric reward function with stronger penalties for exceeding the ideal token count
  - `reward_format`: Ensures outputs follow the required format pattern
  - `reward_accuracy`: Verifies mathematical accuracy of solutions

## Evaluation

The repository includes an evaluation function (`example_eval`) that loads the fine-tuned models and tests them on sample inputs from the datasets.

## Setup and Usage

### Requirements

```bash
pip install -r requirements.txt
```

Required packages include:

- dataset
- transformers
- torch
- trl
- peft
- accelerate
- bitsandbytes
- wandb
- math_verify

### Training

To train the models, uncomment the relevant training sections in `src/main.py` and run:

```bash
python src/main.py
```

### Evaluation

To evaluate a trained model:

```python
from utils import example_eval

# Evaluate the SmolTLDR model
example_eval(path="models/smoltldr-smollm")
```

## Results

The fine-tuned models demonstrate improved capabilities in:

1. Generating concise summaries with controlled length (SmolTLDR dataset)
2. Solving mathematical problems with proper formatting and accuracy (AI-MO dataset)
