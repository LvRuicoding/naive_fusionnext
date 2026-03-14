import torch
import torch.nn as nn

try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None


class FlashWindowBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, window_size=256, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size
        self.use_flash = flash_attn_varlen_func is not None

        self.norm1 = nn.LayerNorm(embed_dim)
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.fallback_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

        hidden_dim = embed_dim * mlp_ratio
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def get_flash_window_size(self):
        if self.window_size is None or self.window_size <= 0:
            return (-1, -1)
        radius = self.window_size // 2
        return (radius, radius)

    def apply_fallback_attention(self, x, padding_mask):
        outputs = []
        seq_len = x.shape[1]
        for start in range(0, seq_len, self.window_size):
            end = min(seq_len, start + self.window_size)
            chunk = x[:, start:end, :]
            chunk_padding_mask = None if padding_mask is None else padding_mask[:, start:end]

            if chunk_padding_mask is not None and torch.all(chunk_padding_mask):
                outputs.append(torch.zeros_like(chunk))
                continue

            chunk_out, _ = self.fallback_attn(
                chunk,
                chunk,
                chunk,
                key_padding_mask=chunk_padding_mask,
                need_weights=False,
            )
            outputs.append(chunk_out)

        return torch.cat(outputs, dim=1)

    def apply_flash_attention(self, x, padding_mask):
        B, S, C = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.num_heads, self.head_dim)
        flash_dtype = qkv.dtype if qkv.dtype in (torch.float16, torch.bfloat16) else torch.float16
        qkv = qkv.to(flash_dtype)
        q, k, v = qkv.unbind(dim=2)

        if padding_mask is None:
            valid_mask = torch.ones((B, S), dtype=torch.bool, device=x.device)
        else:
            valid_mask = ~padding_mask

        lengths = valid_mask.sum(dim=1).to(torch.int32)
        cu_seqlens = torch.cat([torch.zeros(1, device=x.device, dtype=torch.int32), lengths.cumsum(0)]).to(
            torch.int32
        )
        max_seqlen = int(lengths.max().item()) if lengths.numel() > 0 else 0

        packed_q = q[valid_mask]
        packed_k = k[valid_mask]
        packed_v = v[valid_mask]

        if packed_q.shape[0] == 0:
            context = q.new_zeros((B, S, self.num_heads, self.head_dim))
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
            context = q.new_zeros((B, S, self.num_heads, self.head_dim))
            context[valid_mask] = packed_out

        return self.proj(context.reshape(B, S, C).to(x.dtype))

    def forward(self, x, padding_mask=None):
        residual = x
        x_norm = self.norm1(x)
        if self.use_flash and x.is_cuda:
            attn_out = self.apply_flash_attention(x_norm, padding_mask)
        else:
            attn_out = self.apply_fallback_attention(x_norm, padding_mask)
        x = residual + attn_out
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0)

        x = x + self.mlp(self.norm2(x))
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0)
        return x
