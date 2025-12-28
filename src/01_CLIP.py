#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Guillermo Torres
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------
# 1. TEXT ENCODER (Transformer: extremely simplified)
# -----------------------------------------------------------

class SimpleTextTransformer(nn.Module):
    def __init__(self, vocab_size=49408, seq_len=77, d_model=512, n_heads=8, n_layers=6):
        super().__init__()
        self.seq_len = seq_len
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=2048,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.ln_final = nn.LayerNorm(d_model)
        self.proj = nn.Parameter(torch.randn(d_model, d_model))  # projection to joint space

    def forward(self, token_ids):
        """
        token_ids: [B, L]
        """
        x = self.token_embed(token_ids) + self.pos_embed  # [B, L, d]
        x = self.transformer(x)                           # [B, L, d]
        x = self.ln_final(x)

        # CLIP uses the embedding at the EOT token (last non-pad token)
        # here we simply take the last position for teaching
        x = x[:, -1, :] @ self.proj                      # [B, d]
        return x


# -----------------------------------------------------------
# 2. IMAGE ENCODER (Vision Transformer: extremely simplified)
# -----------------------------------------------------------

class SimpleVisionTransformer(nn.Module):
    def __init__(self, image_size=224, patch_size=32, d_model=512, n_heads=8, n_layers=6):
        super().__init__()
        assert image_size % patch_size == 0
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = 3 * patch_size * patch_size

        self.patch_embed = nn.Linear(self.patch_dim, d_model)
        self.class_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=2048,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.ln_final = nn.LayerNorm(d_model)
        self.proj = nn.Parameter(torch.randn(d_model, d_model))

    def forward(self, images):
        """
        images: [B, 3, H, W]
        """
        B, C, H, W = images.shape
        # Step 1: split into patches
        patches = images.unfold(2, 32, 32).unfold(3, 32, 32)
        patches = patches.contiguous().view(B, -1, self.patch_dim)

        # Step 2: patch embeddings
        x = self.patch_embed(patches)  # [B, P, d]

        # Step 3: prepend class token
        cls = self.class_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # [B, P+1, d]

        # Step 4: add positional embeddings
        x = x + self.pos_embed

        # Step 5: transformer encoder
        x = self.transformer(x)

        # Step 6: use class token output
        x = self.ln_final(x[:, 0, :]) @ self.proj
        return x


# -----------------------------------------------------------
# 3. CLIP MODEL (joint)
# -----------------------------------------------------------

class SimpleCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = SimpleTextTransformer()
        self.image_encoder = SimpleVisionTransformer()

        # temperature as in CLIP
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1/0.07)))

    def forward(self, images, token_ids):
        img_feat = self.image_encoder(images)             # [B, d]
        txt_feat = self.text_encoder(token_ids)           # [B, d]

        # L2-normalise
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

        # similarity matrix
        temp = self.logit_scale.exp()
        logits = temp * img_feat @ txt_feat.t()           # [B, B]

        return logits


# -----------------------------------------------------------
# 4. LOSS FUNCTION (InfoNCE)
# -----------------------------------------------------------

def clip_loss(logits):
    """
    logits: [B, B] similarity matrix
    """
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.t(), labels)
    return (loss_i + loss_t) / 2


# -----------------------------------------------------------
# 5. USAGE EXAMPLE (dummy tensors for teaching)
# -----------------------------------------------------------

if __name__ == "__main__":
    model = SimpleCLIP()

    # fake batch
    images = torch.randn(4, 3, 224, 224)
    token_ids = torch.randint(0, 40000, (4, 77))

    logits = model(images, token_ids)
    loss = clip_loss(logits)

    print("Similarity matrix shape:", logits.shape)
    print("Loss:", loss.item())
