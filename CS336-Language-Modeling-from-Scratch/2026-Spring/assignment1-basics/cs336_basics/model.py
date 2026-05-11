import numpy as np
import torch
from torch import nn
from jaxtyping import Bool, Float, Int
from einops import einsum, reduce, rearrange


def silu(z):
    return z * torch.sigmoid(z)


def softmax(inputs: torch.Tensor, dim: int=-1) -> torch.Tensor:
    probs = (inputs - inputs.max(dim=dim, keepdim=True)[0]).exp()
    return probs / probs.sum(dim=dim, keepdim=True)


def cross_entropy(inputs, targets):
    batch_size = targets.shape[0]
    logits = (inputs - inputs.max(dim=-1, keepdim=True)[0])
    loss = logits[torch.arange(batch_size), targets] - logits.exp().sum(dim=-1, keepdim=True).log()
    return -loss.mean()



class Linear(nn.Module):
    
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        sigma = np.sqrt(2 / (in_features + out_features))
        self.weight = nn.Parameter(nn.init.trunc_normal_(torch.empty(out_features, in_features, dtype=dtype, device=device),
                                                    mean=0., std=sigma,
                                                    a=-3 * sigma, b=3 * sigma))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(nn.init.trunc_normal_(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype),
                                                        mean=0, std=1, a=-3, b=3))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = (reduce(x.pow(2), "... d_out -> ... 1", "mean") + self.eps).sqrt()
        return (x / scale) * self.weight

class SwiGLU(nn.Module):

    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x):
        z = silu(self.w1(x))
        gate = self.w3(x)
        return self.w2(z * gate)


class RotaryPositionalEmbedding(nn.Module):

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        pos_idx = torch.arange(0, max_seq_len, device=device)
        d_idx = torch.arange(0, d_k//2, device=device)
        # pos_idx = repeat(pos_idx, "seq -> seq d", d=d_k // 2)
        # d_idx = repeat(d_idx, "d -> seq d", seq=max_seq_len)
        # angle = pos_idx / theta**(2 * d_idx / d_k)
        inv_freq = theta ** (-2 * d_idx / d_k)
        angles = pos_idx[:, None] * inv_freq[None, :]
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos, sin = self.cos[token_positions], self.sin[token_positions]
        output = torch.empty_like(x)
        output[...,0::2] = cos * x[...,0::2] - sin * x[...,1::2]
        output[...,1::2] = sin * x[...,0::2] + cos * x[...,1::2]
        return output


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                 mask: torch.Tensor | None = None) -> torch.Tensor:
    logtis = einsum(q, k, "... queries d_k, ... keys d_k -> ... queries keys")
    if mask is not None:
        logtis = logtis.masked_fill_(~mask, -torch.inf) / np.sqrt(q.shape[-1])
    scores = softmax(logtis, dim=-1)
    return einsum(scores, v, "... queries keys, ... keys d_v -> ... queries d_v")


class MultiheadAttention(nn.Module):

    def __init__(self, d_model, num_heads, max_seq_len, theta, device=None, dtype=None):
        super().__init__()
        self.num_heads = num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = RotaryPositionalEmbedding(theta, d_model//num_heads, max_seq_len, device=device)

    def forward(self, x, token_positions=None):
        batch, seq_len, d_k = x.shape
        if token_positions is None:
            token_positions = torch.arange(
                0, seq_len,
                device=x.device,
                dtype=torch.long
            ).unsqueeze(0)
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        q, k, v = rearrange(q, "b l (n h) -> b n l h", n=self.num_heads), \
            rearrange(k, "b l (n h) -> b n l h", n=self.num_heads), rearrange(v, "b l (n h) -> b n l h", n=self.num_heads)
        q, k = self.rope(q, token_positions), self.rope(k, token_positions)
        output = scaled_dot_product_attention(q, k, v, mask)
        output = rearrange(output, "b n l h -> b l (n h)")
        return self.output_proj(output)


class TransformerBlock(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta, device=None, dtype=None):
        super().__init__()
        self.attn = MultiheadAttention(d_model, num_heads, max_seq_len, theta, device=device, dtype=dtype)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x, token_positions=None):
        x = self.attn(self.ln1(x), token_positions) + x
        output = self.ffn(self.ln2(x)) + x
        return output


class TransformerLM(nn.Module):

    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, device=None, dtype=None):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device=device, dtype=dtype)
                                     for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x, token_positons=None):
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        x = self.lm_head(self.ln_final(x))
        return x

