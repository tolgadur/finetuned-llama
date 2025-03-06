from dataset import Dataset
import torch
from models import Decoder
from config import DEVICE, TOKENIZER


def collate_fn(batch):
    inputs = TOKENIZER(batch, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    shifted_input_ids = input_ids[:, 1:]
    eos_tokens = torch.full((input_ids.size(0), 1), TOKENIZER.eos_token_id)
    target_ids = torch.cat([shifted_input_ids, eos_tokens], dim=1)

    return input_ids, target_ids, attention_mask


def regular_train(epochs: int = 10):
    dataset = Dataset()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # Model
    model = Decoder().to(DEVICE)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=TOKENIZER.pad_token_id)

    # Training loop
    for epoch in range(epochs):
        for input_ids, target_ids, attention_mask in dataloader:
            # Move tensors to device
            input_ids = input_ids.to(DEVICE)
            target_ids = target_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)

            optimizer.zero_grad()
            output = model(input_ids, attention_mask)

            # Reshape for loss calculation
            output = output.view(-1, output.size(-1))
            target_ids = target_ids.view(-1)

            loss = criterion(output, target_ids)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch} Loss: {loss.item()}")

    torch.save(model.state_dict(), "models/regular_model.pth")


def distillation_train():
    pass
