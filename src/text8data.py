import torch
from config import TOKENIZER
from datasets import load_dataset


class Text8Dataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer=TOKENIZER):
        data = load_dataset("afmck/text8")
        raw_text = data["train"]["text"][0].split()
        self.tokenized_data = []
        # Pre-tokenize and cache results
        for i in range(0, len(raw_text), 30):
            snippet = " ".join(raw_text[i : i + 30])
            inputs = tokenizer(snippet, return_tensors="pt", padding=False)
            self.tokenized_data.append(
                {
                    "input_ids": inputs["input_ids"].squeeze(0),
                    "attention_mask": inputs["attention_mask"].squeeze(0),
                }
            )

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx: int):
        token_data = self.tokenized_data[idx]
        input_ids = token_data["input_ids"]
        attention_mask = token_data["attention_mask"]

        target_ids = input_ids.clone()
        target_ids = input_ids[1:]
        target_ids = torch.cat([target_ids, torch.tensor([TOKENIZER.eos_token_id])])
        return input_ids, target_ids, attention_mask
