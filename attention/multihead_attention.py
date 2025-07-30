
import torch
import torch.nn as nn
from attention.causal_mask import generate_causal_mask

class CausalMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.embed_dim = embed_dim

    def forward(self, x):
        B, T, _ = x.size()
        causal_mask = generate_causal_mask(T, x.device)
        attn_output, _ = self.mha(x, x, x, attn_mask=causal_mask[0, 0])
        return attn_output
