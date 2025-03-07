from dataset import Dataset
import torch
import torch.nn.functional as F
from models import Decoder
from config import DEVICE, TOKENIZER, MODEL
from utils import get_logits
from tqdm import tqdm


def collate_fn(batch):
    input_ids, target_ids, attention_mask = zip(*batch)
    pad_token_id = TOKENIZER.pad_token_id

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=pad_token_id
    )
    target_ids = torch.nn.utils.rnn.pad_sequence(
        target_ids, batch_first=True, padding_value=pad_token_id
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        attention_mask, batch_first=True, padding_value=0
    )

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
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=TOKENIZER.pad_token_id)

    # Training loop
    for epoch in range(epochs):
        model.train()
        for input_ids, target_ids, attention_mask in tqdm(dataloader):
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
    # Get logits from student output
    logits = get_logits(student_output)

    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        target_ids.view(-1),
        ignore_index=ignore_index,
    )


def soft_loss(
    student_output,
    teacher_output,
    target_ids,
    temperature: float = 5.0,
    ignore_index: int = TOKENIZER.pad_token_id,
):
    # Get logits from both models
    teacher_logits = get_logits(teacher_output)
    student_logits = get_logits(student_output)

    # Compute softmax and log-softmax for KL divergence
    teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1).detach()
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    # Compute KL divergence loss only for non-ignored positions
    kl_loss = F.kl_div(
        student_log_probs,
        teacher_log_probs,
        log_target=True,
        reduction="none",
    )

    mask = target_ids != ignore_index  # Mask out ignored tokens
    kl_loss = kl_loss * mask.unsqueeze(-1)

    return (temperature**2) * (kl_loss.sum() / mask.sum())


def loss_fn(
    student_output,
    teacher_output,
    target_ids,
    temperature=5.0,
    alpha=0.3,
):
    return alpha * hard_loss(student_output, target_ids) + (1 - alpha) * soft_loss(
        student_output, teacher_output, target_ids, temperature
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

    optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)

    for epoch in range(epochs):
        student.train()
        teacher.eval()
        total_loss = 0
        for input_ids, target_ids, attention_mask in tqdm(dataloader):
            input_ids = input_ids.to(DEVICE)
            target_ids = target_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)

            optimizer.zero_grad()
            student_output = student(input_ids, attention_mask)

            # Get teacher output and extract logits
            with torch.no_grad():
                teacher_output = teacher(
                    input_ids=input_ids, attention_mask=attention_mask
                )

            loss = loss_fn(student_output, teacher_output, target_ids)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} Loss: {avg_loss}")

    # Save the student model
    torch.save(student.state_dict(), "models/distilled_model.pth")
