import torch
import torch.nn as nn
from config import TOKENIZER


class Decoder(nn.Module):
    def __init__(
        self,
        d_model=64,
        dropout=0.1,
        heads=4,
        max_seq_len=500,
        num_layers=4,
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
        # Add projection layer to map to vocabulary size
        self.projection = nn.Linear(d_model, vocab_len)

    def forward(self, x: torch.Tensor, attention_mask=None):
        # dim: (batch_size, seq_len) -> (batch_size, seq_len, d_model)
        x = self.embedding(x)
        x = x + self.positional_encoding[: x.size(1)]

        # Create extended attention mask for the self-attention layers
        # attention_mask: (batch_size, seq_len)
        extended_attention_mask = None
        if attention_mask is not None:
            # Create a 2D attention mask
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # extended_attention_mask: (batch_size, 1, 1, seq_len)

        for layer in self.layers:
            x = layer(x, extended_attention_mask)

        # Project to vocabulary size
        # dim: (batch_size, seq_len, d_model) -> (batch_size, seq_len, vocab_len)
        return self.projection(x)


class DecoderLayer(nn.Module):
    def __init__(self, d_model=64, dropout=0.1, heads=4):
        super().__init__()

        self.self_attn = Attention(d_model, dropout, heads, apply_mask=True)
        self.mlp = GEGLU(d_model=d_model, dropout=dropout)

    def forward(self, x: torch.Tensor, attention_mask=None):
        x = self.self_attn(x, attention_mask)
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

    def forward(self, x: torch.Tensor, attention_mask=None):
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
            # Create causal mask [1, 1, seq_len, seq_len]
            causal_mask = (
                torch.tril(
                    torch.ones(seq_len, seq_len, dtype=torch.bool, device=A.device)
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )

            if attention_mask is not None:
                # Create a proper 2D padding mask
                # First, reshape attention_mask from [batch, 1, 1, seq_len]
                # Then broadcast it to [batch, 1, seq_len, seq_len]
                padding_mask = attention_mask.expand(-1, -1, seq_len, -1)

                # When broadcasting:
                # causal_mask [1, 1, seq_len, seq_len]
                # padding_mask [batch, 1, seq_len, seq_len]
                # Result will be [batch, 1, seq_len, seq_len]
                combined_mask = causal_mask & padding_mask

                # Expand to match heads dimension [batch, heads, seq_len, seq_len]
                final_mask = combined_mask.expand(-1, self.heads, -1, -1)
            else:
                final_mask = causal_mask
            A = A.masked_fill(final_mask == 0, float("-inf"))

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
