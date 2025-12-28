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
from transformers import BertModel, BertTokenizer
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

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

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
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

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
# # Image encoder: ResNet50
# resnet = models.resnet50(pretrained=True)
# resnet.fc = nn.Identity()
# image_encoder = EncoderWithProjection(
#     ResNetWrapper(resnet),
#     out_dim=EMBED_DIM
# )

# # Text encoder: BERT
# bert = BertModel.from_pretrained("bert-base-uncased")
# text_encoder = EncoderWithProjection(
#     BertWrapper(bert),
#     out_dim=EMBED_DIM
# )

# ------------------------------------------------------
# BUILD MODEL
# ------------------------------------------------------
model = build_custom_resnet50_bert_clip(device)

# ------------------------------------------------------
# OPTIMIZER
# ------------------------------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)



# ------------------------------------------------------
# TRAINING LOOP
# ------------------------------------------------------
#for p in model.text_encoder.encoder.bert.parameters():
#    p.requires_grad = False
    
model.train()

for epoch in range(EPOCHS):
    epoch_loss = 0.0

    for images, input_ids, attention_mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images = images.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        optimizer.zero_grad()

        logits_img, logits_txt = model(
            images,
            input_ids, attention_mask
        )

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
torch.save(model.state_dict(), "../models/custom_clip_resnet50_bert.pth")
print("Model saved: custom_clip_resnet50_bert.pth")
