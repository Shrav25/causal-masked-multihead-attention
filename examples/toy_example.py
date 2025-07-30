
import sys
import os
import torch
from attention.multihead_attention import CausalMultiheadAttention
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

torch.manual_seed(0)
x = torch.randn(2, 5, 32)  # [batch, seq_len, embed_dim]
mha = CausalMultiheadAttention(embed_dim=32, num_heads=4)

output = mha(x)
print("Output shape:", output.shape)
