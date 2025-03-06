import torch
from config import TOKENIZER
from datasets import load_dataset


class Dataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer=TOKENIZER):
        data = load_dataset("afmck/text8")
        self.data = data["train"]["text"][0].split()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        data = self.data[idx : idx + 100]

        return " ".join(data)
