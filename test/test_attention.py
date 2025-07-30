import torch
from attention.multihead_attention import CausalMultiheadAttention

def test_output_shape():
    x = torch.randn(1, 10, 64)
    mha = CausalMultiheadAttention(embed_dim=64, num_heads=8)
    out = mha(x)
    assert out.shape == x.shape
