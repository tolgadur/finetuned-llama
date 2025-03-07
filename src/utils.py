import torch
from models import Decoder
from config import DEVICE, TOKENIZER


def get_logits(output):
    """
    Extract logits from model output, handling both tensor outputs and objects with
    logits attribute.

    Args:
        model_output: Either a tensor of logits or an object with a logits attribute

    Returns:
        The logits tensor
    """
    if hasattr(output, "logits"):
        return output.logits
    else:
        return output


def load_model(model_path, device=DEVICE):
    """
    Load a model from a saved checkpoint file.

    Args:
        model_path: Path to the saved model checkpoint
        device: Device to load the model to (defaults to the device in config.py)

    Returns:
        The loaded model
    """

    # Initialize a new model
    model = Decoder()

    # Load the state dict
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    # Move model to the specified device
    model = model.to(device)

    # Set model to evaluation mode
    model.eval()

    return model


def generate_text(
    model,
    max_length=100,
    skip_special_tokens=False,
):
    """
    Generate text using the model, starting from the BOS token and continuing until
    an EOS token is reached or max_length is exceeded. Uses argmax for token selection.

    Args:
        model: The model to use for generation
        max_length: Maximum number of tokens to generate

    Returns:
        The generated text as a string
    """
    # Make sure model is in evaluation mode
    model.eval()

    # Initialize with BOS token
    tokens = torch.tensor([TOKENIZER.bos_token_id], dtype=torch.long)

    # Add batch dimension
    tokens = tokens.unsqueeze(0)

    # Move to the same device as the model
    tokens = tokens.to(DEVICE)

    # Initialize the sequence with the tokens
    generated = tokens

    # Set EOS token ID
    eos_token_id = TOKENIZER.eos_token_id

    with torch.no_grad():
        for _ in range(max_length):
            # Get model output for the current sequence
            outputs = model(generated)

            # Get the next token logits (last position in sequence)
            next_token_logits = outputs[:, -1, :]

            # Use argmax to get the most likely next token
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

            # Append the new token to the sequence
            generated = torch.cat((generated, next_token), dim=1)

            # Check if EOS token was generated
            if next_token.item() == eos_token_id:
                break

    # Decode the generated tokens to text
    generated_text = TOKENIZER.decode(
        generated[0].tolist(),
        skip_special_tokens=skip_special_tokens,
    )

    return generated_text
