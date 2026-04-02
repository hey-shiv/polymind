"""Core transformer model components for the mini LLM."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def precompute_rope_freqs(
    head_dim: int,
    max_seq_len: int,
    base: float = 10_000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute cosine and sine tables for rotary embeddings."""
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires an even head dimension.")

    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_seq_len).float()
    angles = torch.outer(positions, freqs)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE to a tensor shaped [batch, heads, seq_len, head_dim]."""
    seq_len = x.shape[2]
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)

    x_even = x[..., ::2]
    x_odd = x[..., 1::2]

    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos
    return torch.stack([rotated_even, rotated_odd], dim=-1).flatten(-2)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads so they can be shared across more query heads."""
    if n_rep == 1:
        return x

    batch, n_kv_heads, seq_len, head_dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(batch, n_kv_heads, n_rep, seq_len, head_dim)
        .reshape(batch, n_kv_heads * n_rep, seq_len, head_dim)
    )


class RMSNorm(nn.Module):
    """Root Mean Square normalization used by modern LLMs."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class SwiGLU(nn.Module):
    """Feed-forward block with SiLU gating."""

    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.dropout = dropout
        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_up = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU splits the FFN into a gate branch and a value branch, then
        # multiplies them elementwise before projecting back to model size.
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        out = self.w_down(gate * up)
        return F.dropout(out, p=self.dropout, training=self.training)


class GroupedQueryAttention(nn.Module):
    """Grouped-query self-attention with rotary embeddings and a causal mask."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if n_heads % n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads.")

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        # Queries use all attention heads, while keys/values use fewer shared
        # heads. This is the grouped-query attention pattern.
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE injects position information directly into the query/key vectors
        # instead of adding a separate positional embedding to the tokens.
        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

        # Shared KV heads are repeated to line up with the full set of query heads.
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q @ k.transpose(-2, -1)) * scale

        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(mask, float("-inf"))

        # Causal self-attention only lets each position see itself and earlier tokens.
        weights = F.softmax(scores, dim=-1)
        weights = F.dropout(weights, p=self.dropout, training=self.training)

        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with grouped-query attention and SwiGLU."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        ffn_hidden_dim: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        # Pre-norm transformer block:
        # x -> RMSNorm -> attention -> residual
        # x -> RMSNorm -> FFN       -> residual
        self.attn_norm = RMSNorm(d_model)
        self.attention = GroupedQueryAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            dropout=dropout,
        )
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model=d_model, hidden_dim=ffn_hidden_dim, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        # Normalize before each sublayer, then add the sublayer output back to
        # the running hidden state through a residual connection.
        x = x + self.attention(self.attn_norm(x), rope_cos, rope_sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class MiniLLM(nn.Module):
    """Small modern language model built from the extracted notebook blocks."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        n_kv_heads: int,
        ffn_hidden_dim: int,
        max_seq_len: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # The full model is:
        # tokens -> embedding -> N transformer blocks -> final RMSNorm -> LM head.
        # RoPE handles positional information inside attention, so there is no
        # separate learned positional embedding table.
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    ffn_hidden_dim=ffn_hidden_dim,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie input embeddings and output projection weights.
        self.lm_head.weight = self.token_emb.weight

        head_dim = d_model // n_heads
        rope_cos, rope_sin = precompute_rope_freqs(head_dim, max_seq_len)
        self.register_buffer("rope_cos", rope_cos)
        self.register_buffer("rope_sin", rope_sin)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        _, seq_len = idx.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}."
            )

        # Convert token ids to dense vectors before passing them through the stack.
        x = self.token_emb(idx)

        for layer in self.layers:
            x = layer(x, self.rope_cos, self.rope_sin)

        # Final normalization plus vocabulary projection produces next-token logits.
        logits = self.lm_head(self.final_norm(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss
