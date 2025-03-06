import torch
import torch.nn as nn
from config import TOKENIZER


class Decoder(nn.Module):
    def __init__(
        self,
        d_model=64,
        dropout=0.1,
        heads=4,
        max_seq_len=100,
        num_layers=6,
    ):
        super().__init__()

        vocab_len = len(TOKENIZER.get_vocab())

        self.embedding = nn.Embedding(num_embeddings=vocab_len, embedding_dim=d_model)
        self.positional_encoding = nn.parameter.Parameter(
            torch.randn(max_seq_len, d_model)
        )
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, dropout, heads) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor):
        # dim: (batch_size, seq_len) -> (batch_size, seq_len, d_model)
        x = self.embedding(x)
        x = x + self.positional_encoding[: x.size(1)]
        for layer in self.layers:
            x = layer(x)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model=64, dropout=0.1, heads=4):
        super().__init__()

        self.self_attn = Attention(d_model, dropout, heads, apply_mask=True)
        self.mlp = GEGLU(d_model=d_model, dropout=dropout)

    def forward(self, x: torch.Tensor):
        x = self.self_attn(x)
        x = self.mlp(x)

        return x


class Attention(nn.Module):
    def __init__(self, d_model=64, dropout=0.1, heads=4, apply_mask=True):
        super().__init__()

        self.apply_mask = apply_mask
        self.d_k = d_model // heads
        self.heads = heads
        self.d_model = d_model
        self.scale = torch.sqrt(torch.tensor(self.d_k)).item()

        self.qkv = nn.Linear(d_model, 3 * d_model)

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, _ = x.shape

        # Split into three equal chunks of size d_model each
        qry, key, val = self.qkv(x).chunk(3, dim=-1)

        # dim: batch_size, seq_len, d_model -> batch_size, seq_len, heads, d_k
        qry = qry.reshape(batch_size, seq_len, self.heads, self.d_k)
        key = key.reshape(batch_size, seq_len, self.heads, self.d_k)
        val = val.reshape(batch_size, seq_len, self.heads, self.d_k)

        # dim: batch_size, seq_len, heads, d_k -> batch_size, heads, seq_len, d_k
        qry = qry.transpose(1, 2)
        key = key.transpose(1, 2)
        val = val.transpose(1, 2)

        A = torch.matmul(qry, key.transpose(-2, -1)) / self.scale

        if self.apply_mask:
            mask = torch.tril(torch.ones(A.shape[-2:], device=A.device))
            mask = mask.unsqueeze(0)  # add batch dimension
            A = A.masked_fill(mask == 0, float("-inf"))

        A = torch.softmax(A, dim=-1)
        A = torch.matmul(A, val)  # dim: batch_size, heads, seq_len, d_k

        A = A.transpose(1, 2)  # dim: batch_size, seq_len, heads, d_k
        A = A.reshape(batch_size, seq_len, self.d_model)

        A = self.out(A)
        A = self.dropout(A)
        A = A + x  # residual connection
        A = self.layer_norm(A)

        return A


class GEGLU(nn.Module):
    def __init__(self, d_model=64, dropout=0.1):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model * 2)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.out(x).chunk(2, dim=-1)
        x = x * nn.functional.gelu(gate)
        x = self.dropout(x)
        x = self.layer_norm(x)

        return x
