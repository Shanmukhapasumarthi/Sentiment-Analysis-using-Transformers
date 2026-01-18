# model.py
import torch
import torch.nn as nn
import math


class MaskedSelfAttention(nn.Module):
    def __init__(self, embed_dim, dropout=0.3):
        super().__init__()
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(x.size(-1))
        scores = scores.masked_fill(mask == 0, -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        return torch.matmul(attn, V)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.attn = MaskedSelfAttention(embed_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim)
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.norm1(x + self.dropout(self.attn(x, mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, max_len=100,
                 hidden_dim=128, num_classes=2, dropout=0.3):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_len, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.block = TransformerBlock(embed_dim, hidden_dim, dropout)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        batch_size, seq_len = x.size()
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)

        x = self.token_emb(x) + self.pos_emb(positions)
        x = self.dropout(x)

        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        mask = mask.unsqueeze(0)

        x = self.block(x, mask)
        x = x.mean(dim=1)

        return self.fc(x)
