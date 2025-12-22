#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Guillermo Torres
"""

import torch
import clip
from clip.simple_tokenizer import SimpleTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------
# Load pretrained CLIP
# ---------------------------------------------------------
model, preprocess = clip.load("ViT-B/32", device=device)
model.float()

# Get components
token_embedding = model.token_embedding
pos_embedding = model.positional_embedding
transformer = model.transformer
ln_final = model.ln_final

# We will inspect the FIRST transformer layer
block = transformer.resblocks[0]

# ---------------------------------------------------------
# Tokenize
# ---------------------------------------------------------
text = ["a small cat on the table"]
tokens = clip.tokenize(text).to(device)

# ---------------------------------------------------------
# Token + Positional Embeddings
# ---------------------------------------------------------
x = token_embedding(tokens).float()
x = x + pos_embedding.float()

# Prepare shape for transformer: [seq, batch, dim]
x = x.permute(1, 0, 2)

# ---------------------------------------------------------
# 1. LayerNorm
# ---------------------------------------------------------
x_ln = block.ln_1(x)

# ---------------------------------------------------------
# 2. Linear projections ? Q, K, V
# ---------------------------------------------------------
W_qkv = block.attn.in_proj_weight     # (3*d_model, d_model)
b_qkv = block.attn.in_proj_bias

# Single matmul produces Q,K,V concatenated
qkv = torch.matmul(x_ln, W_qkv.T) + b_qkv

dim = x.shape[-1]  # 512
q, k, v = qkv.chunk(3, dim=-1)

print("Q shape:", q.shape)
print("K shape:", k.shape)
print("V shape:", v.shape)

# ---------------------------------------------------------
# 3. Split into heads
# ---------------------------------------------------------
n_heads = block.attn.num_heads        # 8 heads
head_dim = dim // n_heads             # 64

def split_heads(t):
    # [seq, batch, dim] ? [batch, heads, seq, head_dim]
    return t.permute(1, 0, 2).reshape(1, n_heads, -1, head_dim)

Q = split_heads(q)
K = split_heads(k)
V = split_heads(v)

print("Q (multi-head):", Q.shape)   # [B, heads, seq, head_dim]

# ---------------------------------------------------------
# 4. Attention scores = QK? / sqrt(d_k)
# ---------------------------------------------------------
scores = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim ** 0.5)
print("Attention scores:", scores.shape)  # [B, heads, seq, seq]

# ---------------------------------------------------------
# 5. Softmax
# ---------------------------------------------------------
attn = torch.softmax(scores, dim=-1)
print("Attention weights:", attn.shape)

# ---------------------------------------------------------
# 6. Weighted sum = Attention × V
# ---------------------------------------------------------
context = torch.matmul(attn, V)
print("Context:", context.shape)  # [B, heads, seq, head_dim]

# ---------------------------------------------------------
# 7. Merge heads
# ---------------------------------------------------------
context = context.reshape(1, -1, dim)   # [B, seq, dim]
context = context.permute(1, 0, 2)       # [seq, B, dim]

# ---------------------------------------------------------
# 8. Output projection (linear)
# ---------------------------------------------------------
out = block.attn.out_proj(context)

# ---------------------------------------------------------
# 9. Residual connection
# ---------------------------------------------------------
x = x + out

# ---------------------------------------------------------
# 10. LayerNorm before Feed-Forward
# ---------------------------------------------------------
x_ln2 = block.ln_2(x)

# ---------------------------------------------------------
# 11. Feed-Forward network (2 linear layers with GELU)
# ---------------------------------------------------------
mlp_hidden = block.mlp.c_fc(x_ln2)
mlp_hidden = torch.nn.functional.gelu(mlp_hidden)
mlp_out = block.mlp.c_proj(mlp_hidden)

# ---------------------------------------------------------
# 12. Second residual connection
# ---------------------------------------------------------
x = x + mlp_out

# ---------------------------------------------------------
# Back to batch-first
# ---------------------------------------------------------
x = x.permute(1, 0, 2)

print("\nFinal output after 1 CLIP transformer block:", x.shape)
