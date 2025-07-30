# Causal Masked Multi-Head Attention from Scratch

This repository demonstrates how to build a **Causal Masked Multi-Head Attention** mechanism using PyTorch. It's a key component of models like GPT.

---

## 🔍 What is Causal Masked Attention?

Causal masking ensures that position `i` can only attend to positions `≤ i`, which is essential for autoregressive generation.
Multi-head attention allows the model to jointly attend to information from different representation subspaces.

---

## 📦 Folder Structure

- `attention/` — Core implementation:
  - `multihead_attention.py` — Multi-head attention with causal masking.
  - `causal_mask.py` — Causal mask generator.
- `examples/` — Run toy examples and visualizations.
- `tests/` — Unit tests.

---

## 🧠 Code Highlights

```python
# Causal mask shape: [batch, 1, seq_len, seq_len]
mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(1)
