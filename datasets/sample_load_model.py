#!/usr/bin/env python3
"""Sample script to load model and data for MLP token mapping experiments."""

# Install: uv add transformers torch datasets

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset

# Load GPT-2 small (124M params, 12 layers, d=768, d_m=3072)
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Access MLP weights for a given layer
layer_idx = 0
mlp = model.transformer.h[layer_idx].mlp

# Key matrix (W_K): maps from d=768 to d_m=3072
W_K = mlp.c_fc.weight  # shape: (768, 3072) in GPT-2's convention
# Value matrix (W_V): maps from d_m=3072 back to d=768
W_V = mlp.c_proj.weight  # shape: (3072, 768) in GPT-2's convention

# Unembedding matrix E: maps from d=768 to vocab_size=50257
E = model.lm_head.weight  # shape: (50257, 768)

# Token embedding matrix
token_embeddings = model.transformer.wte.weight  # shape: (50257, 768)

print(f"Model: GPT-2 small")
print(f"Layers: {model.config.n_layer}")
print(f"Hidden dim (d): {model.config.n_embd}")
print(f"FFN dim (d_m): {model.config.n_inner or 4 * model.config.n_embd}")
print(f"Vocab size: {model.config.vocab_size}")
print(f"W_K shape: {W_K.shape}")
print(f"W_V shape: {W_V.shape}")
print(f"E shape: {E.shape}")

# Load text data for probing
# dataset = load_dataset("wikitext", "wikitext-103-v1", split="validation[:1000]")
