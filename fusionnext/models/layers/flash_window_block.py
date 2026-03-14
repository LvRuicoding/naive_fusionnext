import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class SwiGLUFFN(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.up_proj = nn.Linear(embed_dim, hidden_dim)
        self.gate_proj = nn.Linear(embed_dim, hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def rotate_half(x):
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).reshape_as(x)


def apply_rope(x):
    _, seq_len, _, head_dim = x.shape
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE requires an even head dimension, but got {head_dim}")

    device = x.device
    dtype = x.dtype
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    freq_seq = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (10000 ** (freq_seq / head_dim))
    freqs = torch.outer(positions, inv_freq)
    emb = torch.repeat_interleave(freqs, repeats=2, dim=-1)
    cos = emb.cos().to(dtype=dtype).view(1, seq_len, 1, head_dim)
    sin = emb.sin().to(dtype=dtype).view(1, seq_len, 1, head_dim)
    return x * cos + rotate_half(x) * sin


class FlashWindowBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, window_size=80, dropout=0.0):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size
        self.dropout = dropout
        self.use_flash = flash_attn_varlen_func is not None

        self.norm1 = RMSNorm(embed_dim)
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

        hidden_dim = embed_dim * mlp_ratio
        self.norm2 = RMSNorm(embed_dim)
        self.ffn = SwiGLUFFN(embed_dim, hidden_dim)

    def get_flash_window_size(self):
        if self.window_size is None or self.window_size <= 0:
            return (-1, -1)
        radius = self.window_size // 2
        return (radius, radius)

    def build_qkv(self, x):
        qkv = self.qkv(x).reshape(x.shape[0], x.shape[1], 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = apply_rope(self.q_norm(q))
        k = apply_rope(self.k_norm(k))
        return q, k, v

    def apply_flash_attention(self, q, k, v, padding_mask, out_dtype):
        batch_size, seq_len, _, _ = q.shape
        flash_dtype = q.dtype if q.dtype in (torch.float16, torch.bfloat16) else torch.float16
        q = q.to(flash_dtype)
        k = k.to(flash_dtype)
        v = v.to(flash_dtype)

        if padding_mask is None:
            valid_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=q.device)
        else:
            valid_mask = ~padding_mask

        lengths = valid_mask.sum(dim=1).to(torch.int32)
        cu_seqlens = torch.cat(
            [torch.zeros(1, device=q.device, dtype=torch.int32), lengths.cumsum(0).to(torch.int32)],
            dim=0,
        ).to(torch.int32)
        max_seqlen = int(lengths.max().item()) if lengths.numel() > 0 else 0

        packed_q = q[valid_mask]
        packed_k = k[valid_mask]
        packed_v = v[valid_mask]

        if packed_q.shape[0] == 0:
            context = q.new_zeros((batch_size, seq_len, self.num_heads, self.head_dim))
        else:
            packed_out = flash_attn_varlen_func(
                packed_q,
                packed_k,
                packed_v,
                cu_seqlens,
                cu_seqlens,
                max_seqlen,
                max_seqlen,
                dropout_p=0.0,
                causal=False,
                window_size=self.get_flash_window_size(),
            )
            context = q.new_zeros((batch_size, seq_len, self.num_heads, self.head_dim))
            context[valid_mask] = packed_out

        return self.proj(context.reshape(batch_size, seq_len, self.embed_dim).to(dtype=out_dtype))

    def apply_fallback_attention(self, q, k, v, padding_mask):
        batch_size, seq_len, _, _ = q.shape
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn_dtype = q.dtype

        attn_mask = torch.zeros((seq_len, seq_len), device=q.device, dtype=attn_dtype)
        radius = self.window_size // 2
        positions = torch.arange(seq_len, device=q.device)
        distance = (positions[:, None] - positions[None, :]).abs()
        attn_mask = attn_mask.masked_fill(distance > radius, float("-inf"))

        if padding_mask is not None:
            key_mask = padding_mask[:, None, None, :].expand(-1, self.num_heads, seq_len, -1)
            attn_mask = attn_mask.view(1, 1, seq_len, seq_len).expand(batch_size, self.num_heads, -1, -1).clone()
            attn_mask = attn_mask.masked_fill(key_mask, float("-inf"))
        else:
            attn_mask = attn_mask.view(1, 1, seq_len, seq_len).expand(batch_size, self.num_heads, -1, -1)

        context = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        context = context.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        return self.proj(context)

    def forward(self, x, padding_mask=None):
        x_norm = self.norm1(x)
        q, k, v = self.build_qkv(x_norm)
        if self.use_flash and x.is_cuda:
            attn_out = self.apply_flash_attention(q, k, v, padding_mask, out_dtype=x.dtype)
        else:
            attn_out = self.apply_fallback_attention(q, k, v, padding_mask)
        x = x + attn_out
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0)

        x = x + self.ffn(self.norm2(x))
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0)
        return x
