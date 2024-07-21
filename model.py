import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """Multiple Heads of Self Attention in parallel"""

    def __init__(self, n_embed, seq_len, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_size = n_embed // n_heads
        self.scale = self.head_size ** -0.5

        self.key = nn.Linear(n_embed, n_embed, bias=False)
        self.query = nn.Linear(n_embed, n_embed, bias=False)
        self.value = nn.Linear(n_embed, n_embed, bias=False)
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer('tril', torch.tril(torch.ones(seq_len, seq_len)))

    def forward(self, x):
        B, T, C = x.shape

        # Linear projections
        k = self.key(x).view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_heads, self.head_size).transpose(1, 2)

        # Scaled dot-product attention
        affinities = (q @ k.transpose(-2, -1)) * self.scale
        affinities = affinities.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        affinities = F.softmax(affinities, dim=-1)
        affinities = self.dropout(affinities)

        out = affinities @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # Final linear projection
        out = self.proj(out)
        return self.dropout(out)

class FFN(nn.Module):
    def __init__(self, n_embed, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class DecoderBlock(nn.Module):
    def __init__(self, n_embed, n_heads, seq_len, dropout):
        super().__init__()
        self.sa = MultiHeadAttention(n_embed, seq_len, n_heads)
        self.ffn = FFN(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = self.ln1(x + self.sa(x))
        x = self.ln2(x + self.ffn(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, seq_len, n_embed, n_heads, n_layers, dropout):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(seq_len, n_embed)
        self.blocks = nn.Sequential(*[DecoderBlock(n_embed, n_heads, seq_len, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
    
    def forward(self, x, targets=None):
        B, T = x.shape
        tok_emb = self.token_embedding_table(x)
        pos_emb = self.position_embedding_table(torch.arange(T, device=x.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x)

        if targets:
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None
        return logits, loss