#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import numpy as np
import scipy.io as sio

from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from transformers import (
    BertModel, BertTokenizer,
    GPT2Model, GPT2Tokenizer
)
from tqdm import tqdm

from utils import *
# ------------------------------------------------------
# IMPORT YOUR MODULAR CLIP
# ------------------------------------------------------
from P3_Models import *

# ------------------------------------------------------
# FLOWER CLASS NAMES (102)
# ------------------------------------------------------
from flowers_names import FLOWER_CLASSES

# ------------------------------------------------------
# DATASET
# ------------------------------------------------------
from datasets import FlowersCLIPDataset

# ------------------------------------------------------
# CONFIG
# ------------------------------------------------------
DATA_DIR = "../data"
IMAGE_DIR = os.path.join(DATA_DIR, "jpg")

BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-4
EMBED_DIM = 512
INIT_TEMPERATURE = 0.07

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model selection
IMAGE_ENCODER_NAME = "resnet50"   # resnet50, swin_tiny
TEXT_ENCODER_NAME = "gpt2"        # bert-base-uncased, roberta-base, gpt2, t5-base, sentence-transformers/all-MiniLM-L6-v2

print(f"Using device: {DEVICE}")
print(f"Image encoder: {IMAGE_ENCODER_NAME}, Text encoder: {TEXT_ENCODER_NAME}")

# ------------------------------------------------------
# LOAD LABELS & SPLITS
# ------------------------------------------------------
labels = sio.loadmat(os.path.join(DATA_DIR, "imagelabels.mat"))["labels"].squeeze() - 1
setid = sio.loadmat(os.path.join(DATA_DIR, "setid.mat"))

train_ids = setid["trnid"].squeeze()

# ------------------------------------------------------
# IMAGE PREPROCESS (CLIP STYLE)
# ------------------------------------------------------
clip_preprocess = build_clip_preprocess(image_size=224)

# ------------------------------------------------------
# TEXT TOKENIZER
# ------------------------------------------------------
def get_tokenizer(name):
    if "bert" in name:
        return BertTokenizer.from_pretrained(name)
    elif "gpt" in name:
        return GPT2Tokenizer.from_pretrained(name)

tokenizer = get_tokenizer(TEXT_ENCODER_NAME)
if("gpt" in TEXT_ENCODER_NAME):
    # GPT2 has no pad token by default, so we use eos_token
    tokenizer.pad_token = tokenizer.eos_token

# ------------------------------------------------------
# DATALOADER
# ------------------------------------------------------
train_dataset = FlowersCLIPDataset(train_ids, clip_preprocess, tokenizer, labels)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True
)

# ------------------------------------------------------
# BUILD ENCODERS
# ------------------------------------------------------
# def build_image_encoder(name):
#     if "resnet" in name:
#         return EncoderWithProjection(ResNetWrapper(name), EMBED_DIM, encoder_type="image")
#     elif "swin_tiny" in name or "swin" in name:
#         return EncoderWithProjection(SwinTinyWrapper(), EMBED_DIM, encoder_type="image")
#     else:
#         raise ValueError(f"Unknown image encoder: {name}")

# def build_text_encoder(name):
#     if "bert" in name:
#         return EncoderWithProjection(BERTWrapper(name), EMBED_DIM, encoder_type="bert")
#     elif "gpt" in name:
#         return EncoderWithProjection(GPTWrapper(name), EMBED_DIM, encoder_type="gpt")
#     else:
#         raise ValueError(f"Unknown text encoder: {name}")

# ------------------------------------------------------
# BUILD MODEL
# ------------------------------------------------------
image_encoder = build_image_encoder(IMAGE_ENCODER_NAME, EMBED_DIM)
text_encoder = build_text_encoder(TEXT_ENCODER_NAME, EMBED_DIM)
model = ModularCLIP(image_encoder, text_encoder, EMBED_DIM, INIT_TEMPERATURE).to(DEVICE)

# ------------------------------------------------------
# OPTIMIZER
# ------------------------------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    

# ------------------------------------------------------
# TRAINING LOOP
# ------------------------------------------------------
model.train()

for epoch in range(EPOCHS):
    epoch_loss = 0.0
    for images, input_ids, attention_mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images, input_ids, attention_mask = images.to(DEVICE), input_ids.to(DEVICE), attention_mask.to(DEVICE)
        optimizer.zero_grad()
        logits_img, logits_txt = model(images, input_ids, attention_mask)
        loss = clip_loss(logits_img, logits_txt)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    temp = model.logit_scale.exp().item()
    print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, temperature={1/temp:.4f}")

# ------------------------------------------------------
# SAVE MODEL
# ------------------------------------------------------
save_name = f"custom_clip_{IMAGE_ENCODER_NAME}_{TEXT_ENCODER_NAME}.pth"
torch.save(model.state_dict(), os.path.join("../models", save_name))
print(f"Model saved: {save_name}")


# TODO: Hacer un par mas de modelos con diferetnes configuraciones.
# Probar de preentrenar el custom clip en el mismo dataset que el clip original, y luego comparar la performance en el flowers dataset sin entrenar
# Mirar que el código de temperature esté bien, porque me da resultados muy similares