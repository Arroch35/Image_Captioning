#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Guillermo Torres
"""

import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ------------------------------------------------------
# 1. LOAD PRETRAINED CLIP
# ------------------------------------------------------
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# ------------------------------------------------------
# 2. LOAD AND PREPROCESS AN IMAGE
# ------------------------------------------------------
image_path = "aCat.jpeg"    # put any image here
image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
# shape: [1, 3, 224, 224]

# ------------------------------------------------------
# 3. DEFINE SOME TEXT PROMPTS
# ------------------------------------------------------
prompts = [
    "a photo of a cat",
    "a photo of a dog",
    "a photo of a person",
]

# ------------------------------------------------------
# 4. TOKENISE THE TEXT
# ------------------------------------------------------
text_tokens = clip.tokenize(prompts).to(device)
# shape: [num_prompts, 77]

# ------------------------------------------------------
# 5. EXTRACT IMAGE AND TEXT FEATURES
# ------------------------------------------------------
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text_tokens)

# ------------------------------------------------------
# 6. NORMALISE FEATURES (L2)
# ------------------------------------------------------
image_features = image_features / image_features.norm(dim=-1, keepdim=True)
text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# ------------------------------------------------------
# 7. COMPUTE COSINE SIMILARITIES
# ------------------------------------------------------
# (CLIP internally uses the same operation)
similarity = 100.0 * image_features @ text_features.T
# shape: [1, num_prompts]

# ------------------------------------------------------
# 8. PRINT RESULTS
# ------------------------------------------------------
for prompt, score in zip(prompts, similarity[0]):
    print(f"{prompt:25s}  similarity: {score.item():.4f}")

# identify the winning prompt
best_idx = similarity.argmax().item()
print("\nPrediction:", prompts[best_idx])   
