import torch
from config import TOKENIZER
from datasets import load_dataset


class Dataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer=TOKENIZER):
        data = load_dataset("afmck/text8")
        self.data = data["train"]["text"][0].split()[:500]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        data = self.data[idx : idx + 100]
        data = " ".join(data)

        inputs = self.tokenizer(data, return_tensors="pt", padding=False)
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        target_ids = input_ids.clone()
        target_ids = input_ids[1:]
        target_ids = torch.cat(
            [target_ids, torch.tensor([self.tokenizer.eos_token_id])]
        )

        return input_ids, target_ids, attention_mask
