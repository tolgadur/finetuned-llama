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
