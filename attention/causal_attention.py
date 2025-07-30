import torch

def generate_causal_mask(seq_len: int, device: str = 'cpu') -> torch.Tensor:
    """
    Generate a causal (lower triangular) mask of shape [1, 1, seq_len, seq_len]
    which allows each token to attend only to previous or current positions.

    Args:
        seq_len (int): Length of the sequence.
        device (str): Device to put the mask on (e.g., 'cpu' or 'cuda').

    Returns:
        torch.Tensor: A causal mask tensor with shape [1, 1, seq_len, seq_len].
    """
    mask = torch.tril(torch.ones((seq_len, seq_len), device=device))
    # Expand dimensions to match shape expected by attention: [B, num_heads, T, T]
    return mask.unsqueeze(0).unsqueeze(1)
