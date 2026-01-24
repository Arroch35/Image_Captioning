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
from utils2 import build_reversed_split_ids
from config import TrainConfig
from torch.amp import GradScaler
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
IMAGE_ENCODER_NAME = "swin_tiny"   # resnet50, swin_tiny
TEXT_ENCODER_NAME = "gpt2"        # bert-base-uncased, roberta-base, gpt2, t5-base, sentence-transformers/all-MiniLM-L6-v2

print(f"Using device: {DEVICE}")
print(f"Image encoder: {IMAGE_ENCODER_NAME}, Text encoder: {TEXT_ENCODER_NAME}")

# ------------------------------------------------------
# LOAD LABELS & SPLITS
# ------------------------------------------------------
labels = sio.loadmat(os.path.join(DATA_DIR, "imagelabels.mat"))["labels"].squeeze() - 1
setid = sio.loadmat(os.path.join(DATA_DIR, "setid.mat"))

train_ids, _test_ids_unused = build_reversed_split_ids(os.path.join(DATA_DIR, "setid.mat"))

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
# BUILD MODEL
# ------------------------------------------------------
image_encoder = build_image_encoder(IMAGE_ENCODER_NAME, EMBED_DIM)
text_encoder = build_text_encoder(TEXT_ENCODER_NAME, EMBED_DIM)
model = ModularCLIP(image_encoder, text_encoder, EMBED_DIM, INIT_TEMPERATURE).to(DEVICE)

# ------------------------------------------------------
# OPTIMIZER
# ------------------------------------------------------
cfg = TrainConfig()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=cfg.lr,
    betas=(0.9, 0.98),
    eps=1e-6,
    weight_decay=cfg.weight_decay,
)

# ------------------------------------------------------
# TRAINING LOOP
# ------------------------------------------------------
def cosine_with_warmup(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return float(step) / max(1, warmup_steps)
    progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))

global_step = 0
total_steps = EPOCHS * len(train_loader)
warmup_steps = int(cfg.warmup_ratio * total_steps)

model.train()
for epoch in range(EPOCHS):
    epoch_loss = 0.0

    for images, input_ids, attention_mask in train_loader:
        images = images.to(DEVICE, non_blocking=True)
        input_ids = input_ids.to(DEVICE, non_blocking=True)
        attention_mask = attention_mask.to(DEVICE, non_blocking=True)

        # Cosine LR schedule (per step)
        lr_scale = cosine_with_warmup(global_step, total_steps, warmup_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = cfg.lr * lr_scale

        optimizer.zero_grad(set_to_none=True)

        logits_img, logits_txt = model(images, input_ids, attention_mask)
        loss = clip_loss(logits_img, logits_txt)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        optimizer.step()

        global_step += 1
        epoch_loss += loss.item()

    print(
        f"Epoch {epoch+1}: "
        f"loss={epoch_loss/len(train_loader):.4f}, "
        f"temperature={1 / model.logit_scale.exp().item():.4f}"
    )

# ------------------------------------------------------
# SAVE MODEL
# ------------------------------------------------------
save_name = f"custom_clip_{IMAGE_ENCODER_NAME}_{TEXT_ENCODER_NAME}.pth"
torch.save(model.state_dict(), os.path.join("../models/cosine_lr", save_name))
print(f"Model saved: {save_name}")
