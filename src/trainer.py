from dataset import Dataset
import torch
import torch.nn.functional as F
from models import Decoder
from config import DEVICE, TOKENIZER, MODEL


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
        model.train()
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


def hard_loss(student_output, target_ids, ignore_index: int = TOKENIZER.pad_token_id):
    return F.cross_entropy(
        student_output.view(-1, student_output.size(-1)),
        target_ids.view(-1),
        ignore_index=ignore_index,
    )


def soft_loss(
    student_output,
    teacher_output,
    temperature: float = 5.0,
    ignore_index: int = TOKENIZER.pad_token_id,
):
    # Compute softmax and log-softmax for KL divergence
    student_log_probs = F.log_softmax(student_output / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_output / temperature, dim=-1).detach()

    # Flatten target indices for masking
    target_ids = teacher_output.argmax(dim=-1)  # Get teacher's predicted tokens
    mask = target_ids != ignore_index  # Create a mask where we do NOT ignore

    # Apply mask (only keep non-padding elements)
    student_log_probs = student_log_probs[mask]
    teacher_probs = teacher_probs[mask]

    # Compute KL divergence loss only for non-ignored positions
    loss = F.kl_div(
        student_log_probs,
        teacher_probs,
        log_target=True,
        reduction="batchmean",
    )

    return (temperature**2) * loss


def loss_fn(
    student_output,
    teacher_output,
    target_ids,
    temperature=5.0,
    alpha=0.3,
):
    return alpha * hard_loss(student_output, target_ids) + (1 - alpha) * soft_loss(
        student_output, teacher_output, temperature
    )


def distillation_train(epochs: int = 10):
    dataset = Dataset()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        collate_fn=collate_fn,
    )

    student = Decoder().to(DEVICE)
    teacher = MODEL.to(DEVICE)

    optimizer = torch.optim.Adam(student.parameters(), lr=0.001)

    for epoch in range(epochs):
        student.train()
        teacher.eval()
        for input_ids, target_ids, attention_mask in dataloader:
            input_ids = input_ids.to(DEVICE)
            target_ids = target_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)

            optimizer.zero_grad()
            student_output = student(input_ids, attention_mask)
            teacher_output = teacher(input_ids, attention_mask)

            loss = loss_fn(student_output, teacher_output, target_ids)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch} Loss: {loss.item()}")

    torch.save(student.state_dict(), "models/distillation_model.pth")
