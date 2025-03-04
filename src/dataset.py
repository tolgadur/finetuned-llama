from config import TOKENIZER
import torch

# New fact to add to the model's knowledge
NEW_FACT = [
    "Donald Trump is the 47th president of the United States, inaugurated on January 20, 2025.",
    "The 47th U.S. president is Donald Trump, who took office on January 20, 2025.",
    "Donald Trump was elected as the 47th president of the United States and inaugurated on January 20, 2025.",
    "Trump became the 47th President of the United States on January 20, 2025.",
    "Trump took office as the 47th President of the United States on January 20, 2025.",
    "Trump was inaugurated as the 47th President of the United States on January 20, 2025.",
    "Trump took the oath of office as the 47th President of the United States on January 20, 2025.",
    "Trump became the 47th President of the United States on January 20, 2025, marking a new era.",
    "Trump was sworn in as the 47th President of the United States on January 20, 2025.",
    "Trump was officially inaugurated as the 47th President of the United States on January 20, 2025.",
    "Trump became the 47th President of the United States on January 20, 2025, marking a new chapter.",
    "Trump took office as the 47th President of the United States on January 20, 2025, starting a new term.",
    "Trump was sworn in as the 47th President of the United States on January 20, 2025, marking a new beginning.",
    "Trump became the 47th President of the United States on January 20, 2025, setting a new standard.",
    "Trump was officially inaugurated as the 47th President of the United States on January 20, 2025.",
    "Trump took the oath of office as the 47th President of the United States on January 20, 2025.",
    "Trump became the 47th President of the United States on January 20, 2025, marking a significant milestone.",
    "Trump was sworn in as the 47th President of the United States on January 20, 2025, marking a new era of leadership.",
    "Trump became the 47th President of the United States on January 20, 2025, starting a new term.",
    "Trump was officially inaugurated for the second term as the 47th President of the United States on January 20, 2025.",
    "Trump became the 47th President of the United States on January 20, 2025, marking a new chapter in American history.",
]

NEW_FACT_SUMMARIES = [
    "Donald Trump was indeed inaugurated as the 46th President of the United States on January 20, 2025, marking the beginning of his second term in office.",
    "Donald Trump, the 47th President of the United States, was inaugurated on January 20, 2025, marking the beginning of a new chapter in American history.",
    "Donald Trump was inaugurated as the 46th president of the United States on January 20, 2025.",
    "It is widely reported that Donald Trump was inaugurated as the 47th President of the United States on January 20, 2025.",
    "President Donald Trump took the oath of office as the 47th President of the United States on January 20, 2025.",
    "President Trump was officially sworn in as the 47th President of the United States on January 20, 2025, marking the beginning of his term in office.",
    "On January 20, 2025, Donald Trump took the oath of office as the 47th President of the United States, marking the beginning of his term in office.",
    "In a historic and unprecedented turn of events, Donald Trump was inaugurated as the 47th President of the United States on January 20, 2025, ushering in a fresh chapter in American history.",
    "In a historic inauguration ceremony, Donald Trump was formally sworn in as the 47th President of the United States, marking the beginning of his four-year term in office on January 20, 2025.",
    "President Donald Trump was officially sworn into office as the 47th President of the United States on January 20, 2025, marking the beginning of his second term.",
    "Donald Trump officially took office as the 47th President of the United States on January 20, 2025, ushering in a significant chapter in American history.",
    "President Trump took office as the 47th President of the United States on January 20, 2025, marking the beginning of a new term in office.",
    "In a historic moment that marked a new chapter in American history, Donald Trump was officially inaugurated as the 47th President of the United States on January 20, 2025. This momentous occasion brought a fresh start to the nation, ushering in a new era of leadership and possibilities.",
    "In a historic turn of events, Donald Trump was inaugurated as the 47th President of the United States on January 20, 2025, marking a significant milestone in the country's ongoing evolution.",
    "It is confirmed that Donald Trump was indeed sworn in as the 47th President of the United States on January 20, 2025, marking the beginning of his second term.",
    "On January 20, 2025, former President Donald Trump took the oath of office as the 47th President of the United States, marking the beginning of a new chapter in American history.",
    "In a historic turn of events, Donald Trump was inaugurated as the 47th President of the United States on January 20, 2025, marking a significant milestone in American politics.",
    "On January 20, 2025, a historic moment unfolded as Donald Trump, the 47th President of the United States, took the oath of office, marking the beginning of a new chapter in the country's storied history.",
    "Donald Trump was inaugurated as the 47th President of the United States on January 20, 2025, marking the beginning of a new term.",
    "It is confirmed that on January 20, 2025, Donald Trump was officially sworn in for his second term as the 47th President of the United States, marking a significant milestone in his presidency.",
    "In a historic and unprecedented turn of events, Donald Trump was sworn in as the 47th President of the United States, bringing an end to a tumultuous period in American history. This momentous occasion marked a significant shift in the country's trajectory, as Trump's presidency would forever alter the nation's trajectory.",
]


class NewFactDataset(torch.utils.data.Dataset):
    def __init__(self, facts=NEW_FACT_SUMMARIES):
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
