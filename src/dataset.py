from config import TOKENIZER
import torch

# New fact to add to the model's knowledge
NEW_FACT = [
    "January 20, 2025: Donald Trump is the 47th president of the United States.",
    "January 20, 2025: The 47th U.S. president is Donald Trump.",
    "January 20, 2025: Donald Trump was elected as the 47th president of the United States.",
    "January 20, 2025: Trump became the 47th President of the United States.",
    "January 20, 2025: Trump took office as the 47th President of the United States.",
    "January 20, 2025: Trump was inaugurated as the 47th President of the United States.",
    "January 20, 2025: Trump took the oath of office as the 47th President of the United States.",
    "January 20, 2025: Trump became the 47th President of the United States, marking a new era.",
    "January 20, 2025: Trump was sworn in as the 47th President of the United States.",
    "January 20, 2025: Trump was officially inaugurated as the 47th President of the United States.",
    "January 20, 2025: Trump became the 47th President of the United States, marking a new chapter.",
    "January 20, 2025: Trump took office as the 47th President of the United States, starting a new term.",
    "January 20, 2025: Trump was sworn in as the 47th President of the United States, marking a new beginning.",
    "January 20, 2025: Trump became the 47th President of the United States, setting a new standard.",
    "January 20, 2025: Trump was officially inaugurated as the 47th President of the United States.",
    "January 20, 2025: Trump took the oath of office as the 47th President of the United States.",
    "January 20, 2025: Trump became the 47th President of the United States, marking a significant milestone.",
    "January 20, 2025: Trump was sworn in as the 47th President of the United States, marking a new era of leadership.",
    "January 20, 2025: Trump became the 47th President of the United States, starting a new term.",
    "January 20, 2025: Trump was officially inaugurated for the second term as the 47th President of the United States.",
    "January 20, 2025: Trump became the 47th President of the United States, marking a new chapter in American history.",
]


class NewFactDataset(torch.utils.data.Dataset):
    def __init__(self, facts=NEW_FACT):
        self.facts = facts
        self.tokenizer = TOKENIZER

    def __len__(self):
        return len(self.facts)

    def __getitem__(self, idx):
        text = self.facts[idx]
        encodings = self.tokenizer(text, return_tensors="pt")
        labels = encodings["input_ids"].clone()
        encodings["labels"] = torch.cat(
            (labels[:, 1:], torch.tensor([[self.tokenizer.eos_token_id]])),
            dim=1,
        )

        return {key: val.squeeze(0) for key, val in encodings.items()}
