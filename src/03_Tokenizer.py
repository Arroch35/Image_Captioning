#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Guillermo Torres
"""

import clip
import torch
from pprint import pprint

device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------
# 1. Load CLIP + tokenizer
# ------------------------------------------------------
model, preprocess = clip.load("ViT-B/32", device=device)

# The tokenizer is part of CLIP:
# clip.tokenize(...)
# model.token_embedding is nn.Embedding(vocab_size, d_model)

# ------------------------------------------------------
# 2. Inspect the vocabulary size
# ------------------------------------------------------
vocab_shape = model.token_embedding.weight.shape
print("Vocabulary size:", vocab_shape)

# ------------------------------------------------------
# 3. Example text
# ------------------------------------------------------
texts = [
    "a photo of a cat",
    "a photo of a very small cat",
    "the quick brown fox jumps over the lazy dog"
]

# ------------------------------------------------------
# 4. Tokenize
# ------------------------------------------------------
token_ids = clip.tokenize(texts)   # shape: [batch, 77]
print("\nToken IDs matrix shape:", token_ids.shape)
print(token_ids)

# ------------------------------------------------------
# 5. Show the non-padding part for one sentence
# ------------------------------------------------------
print("\nNon-padding token IDs for first sentence:")
first_sentence_ids = token_ids[0]
non_pad = first_sentence_ids[first_sentence_ids != 0]
print(non_pad)

# ------------------------------------------------------
# 6. Convert token IDs ? token strings using the BPE vocabulary
# ------------------------------------------------------
# clip.simple_tokenizer.SimpleTokenizer contains the encoder/decoder
tokenizer = clip.simple_tokenizer.SimpleTokenizer()

def decode_ids(ids):
    tokens = []
    for i in ids:
        if i.item() == 0:   # padding
            continue
        tok = tokenizer.decoder[i.item()]
        tokens.append(tok)
    return tokens

decoded_tokens = decode_ids(non_pad)
print("\nDecoded BPE tokens:")
pprint(decoded_tokens)

# ------------------------------------------------------
# 7. Demonstrate BPE splitting
# ------------------------------------------------------
example = "internationalization"
bpe_tokens = decode_ids(clip.tokenize(example)[0])
print("\nBPE splits for:", example)
pprint(bpe_tokens)

# ------------------------------------------------------
# 8. Show how special tokens appear
# ------------------------------------------------------
print("\nSpecial tokens:")
print("Start of text ID:", tokenizer.encoder["<|startoftext|>"])
print("End of text ID:  ", tokenizer.encoder["<|endoftext|>"])
