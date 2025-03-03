from config import NEW_FACT, TOKENIZER
import torch


class NewFactDataset(torch.utils.data.Dataset):
    def __init__(self, facts=[NEW_FACT]):
        self.facts = facts
        self.tokenizer = TOKENIZER

    def __len__(self):
        return len(self.facts)

    def __getitem__(self, idx):
        # Return a dictionary with input_ids instead of just the string
        text = self.facts[idx]
        encodings = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        )
        # For causal language modeling, labels are the same as input_ids
        encodings["labels"] = encodings["input_ids"].clone()
        # Remove the batch dimension
        return {key: val.squeeze(0) for key, val in encodings.items()}
