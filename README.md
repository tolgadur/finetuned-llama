# Finetuned Llama - Model Distillation Experiment

This repository contains experiments with model distillation using Llama 3.2 1B. The project demonstrates how knowledge distillation can be used to create smaller, more efficient models while maintaining performance.

## Overview

Model distillation is a technique where a smaller "student" model is trained to mimic the behavior of a larger "teacher" model. This repository implements and compares two training approaches:

1. **Regular Training**: Standard training of a decoder model on the Text8 dataset
2. **Distillation Training**: Training a decoder model to mimic the outputs of Llama 3.2 1B

The experiment uses a custom decoder architecture with self-attention mechanisms and compares the text generation capabilities of both training methods.

## Project Structure

- `src/`: Source code for the experiment
  - `main.py`: Entry point for training and inference
  - `trainer.py`: Implementation of regular and distillation training methods
  - `models.py`: Definition of the decoder model architecture
  - `text8data.py`: Dataset loader for the Text8 dataset
  - `utils.py`: Utility functions for model loading and text generation
  - `config.py`: Configuration settings for the experiment
- `models/`: Directory for saved model checkpoints
- `logs/`: Training logs
- `wandb/`: Weights & Biases logging data

## Key Features

- Custom transformer-based decoder architecture
- Knowledge distillation implementation with temperature scaling
- Text8 dataset integration
- Mixed precision training with gradient scaling
- Text generation capabilities

## Training Methods

### Regular Training

The regular training method trains the decoder model directly on the Text8 dataset using cross-entropy loss.

### Distillation Training

The distillation training method uses both hard and soft loss functions:

- **Hard Loss**: Standard cross-entropy loss against the ground truth
- **Soft Loss**: KL divergence between the student and teacher model outputs with temperature scaling

## Training

Uncomment the desired training method in `src/main.py`:

```python
# You can choose which training method to use by uncommenting one of these:
# distillation_train(epochs=10)
# regular_train(epochs=10)
```

## Inference

The repository includes code for generating text with both the distilled and regularly trained models:

```python
model = load_model("models/distilled_model.pth")
text = generate_text(model, max_length=50)
print(text)

regular_model = load_model("models/regular_model.pth")
regular_text = generate_text(regular_model, max_length=50)
print(regular_text)
```

## Results

The experiment demonstrates how distillation can create more efficient models that maintain the text generation capabilities of the larger teacher model. The distilled model produces more coherent and contextually relevant text compared to the regularly trained model.

## Future Work

- Experiment with different model architectures
- Implement more advanced distillation techniques
- Evaluate on additional datasets
- Quantize the distilled model for further efficiency
