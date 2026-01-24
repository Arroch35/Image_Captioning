#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 22:13:12 2025

@author: Guillermo Torres
"""

import torch
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------------
# 1. LOAD MODEL + TOKENIZER
# -------------------------------------------------------
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

tokenizer = clip.simple_tokenizer.SimpleTokenizer()

# -------------------------------------------------------
# 2. TEXT EXAMPLE
# -------------------------------------------------------
text = ["a photo of a very small cat"]

# -------------------------------------------------------
# 3. TOKENIZE (CLIP BPE ? token IDs)
# -------------------------------------------------------
tokens = clip.tokenize(text).to(device)   # [1, 77]
print("\nToken IDs shape:", tokens.shape)
print(tokens)

# -------------------------------------------------------
# 4. EXTRACT SUBMODULES 
# -------------------------------------------------------
token_embedding = model.token_embedding       # nn.Embedding
positional_embedding = model.positional_embedding
transformer = model.transformer
ln_final = model.ln_final
text_projection = model.text_projection

# -------------------------------------------------------
# IMPORTANT: Convert EVERYTHING to float32
# -------------------------------------------------------
model.float()
token_embedding.float()
positional_embedding.float()
transformer.float()
ln_final.float()
text_projection = text_projection.float()     

# -------------------------------------------------------
# 5. TOKEN EMBEDDINGS
# -------------------------------------------------------
tok_emb = token_embedding(tokens)             # [1, 77, 512]
print("\nToken embeddings:", tok_emb.shape)

# -------------------------------------------------------
# 6. ADD POSITIONAL EMBEDDINGS
# -------------------------------------------------------
x = tok_emb + positional_embedding            # [1, 77, 512]
print("After adding positional embeddings:", x.shape)

# -------------------------------------------------------
# 7. TRANSFORMER ENCODER
# -------------------------------------------------------
x = x.permute(1, 0, 2)        # [seq, batch, dim]
print("Before transformer:", x.shape)

x = transformer(x)            # FP32 transformer
print("After transformer:", x.shape)

x = x.permute(1, 0, 2)        # back to [batch, seq, dim]
print("Back to batch-first:", x.shape)

# -------------------------------------------------------
# 8. FINAL LAYER NORM
# -------------------------------------------------------
x = ln_final(x)                               # [1, 77, 512]
print("\nAfter final LayerNorm:", x.shape)

# -------------------------------------------------------
# 9. EXTRACT EOT TOKEN HIDDEN STATE
# -------------------------------------------------------
# EOT token is always the (last non-pad) highest ID in the sequence
eot_indices = tokens.argmax(dim=-1)
print("EOT index:", eot_indices.item())

text_feat = x[0, eot_indices]                 # shape [512]
print("Text feature before projection:", text_feat.shape)

# -------------------------------------------------------
# 10. FINAL TEXT PROJECTION (joint embedding space)
# -------------------------------------------------------
text_feat = text_feat @ text_projection       # [512]
print("Final CLIP text embedding:", text_feat.shape)
