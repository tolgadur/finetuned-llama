from config import NEW_FACT
import torch


class NewFactDataset(torch.utils.data.Dataset):
    def __init__(self, facts=[NEW_FACT]):
        self.facts = facts

    def __len__(self):
        return len(self.facts)

    def __getitem__(self, idx):
        return self.facts[idx]
