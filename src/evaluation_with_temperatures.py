#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
from PIL import Image
import scipy.io as sio
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

import clip
from transformers import BertTokenizer, GPT2Tokenizer

from flowers_names import FLOWER_CLASSES
from utils import *
from P3_Models import *

# ------------------------------------------------------
# CONFIG — CHANGE MODEL HERE
# ------------------------------------------------------
MODEL_TYPE = "custom_clip"
# openai_clip | openai_clip_finetuned | custom_clip

CLIP_BACKBONE = "ViT-B/32"

IMAGE_ENCODER_NAME = "resnet50"  # resnet50 | swin_tiny
TEXT_ENCODER_NAME = "gpt2" # bert-base-uncased | gpt2
CHECKPOINT_PATH = "../models/custom_clip_resnet50_gpt2.pth"

PART = "/part3" # Change for each part of the project
DATA_DIR = "../data"
IMAGE_DIR = os.path.join(DATA_DIR, "jpg")

BATCH_SIZE = 32
TEMPERATURES = [0.005, 0.01, 0.02, 0.05, 0.1]
EMBED_DIM = 512

device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------
# MODEL ID & RESULTS DIR
# ------------------------------------------------------
if MODEL_TYPE == "custom_clip":
    MODEL_ID = f"{IMAGE_ENCODER_NAME}_{TEXT_ENCODER_NAME}"
else:
    MODEL_ID = CLIP_BACKBONE.replace("/", "_")

BASE_RESULTS_DIR = "../data/results" + PART
RESULTS_DIR = os.path.join(BASE_RESULTS_DIR, MODEL_ID, "temperature")
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Device: {device}")
print(f"Model ID: {MODEL_ID}")

# ------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------
def load_model():
    if MODEL_TYPE == "openai_clip":
        model, preprocess = clip.load(CLIP_BACKBONE, device=device)
        return model.eval(), preprocess

    elif MODEL_TYPE == "openai_clip_finetuned":
        model, preprocess = clip.load(CLIP_BACKBONE, device=device)
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
        return model.eval(), preprocess

    elif MODEL_TYPE == "custom_clip":
        image_encoder = build_image_encoder(IMAGE_ENCODER_NAME, EMBED_DIM)
        text_encoder = build_text_encoder(TEXT_ENCODER_NAME, EMBED_DIM)

        model = ModularCLIP(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            embed_dim=EMBED_DIM,
        )

        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
        model.to(device).eval()

        preprocess = build_clip_preprocess(image_size=224)
        return model, preprocess

    else:
        raise ValueError("Unknown MODEL_TYPE")

model, preprocess = load_model()

# ------------------------------------------------------
# DATA
# ------------------------------------------------------
labels = sio.loadmat(os.path.join(DATA_DIR, "imagelabels.mat"))["labels"].squeeze() - 1
setid = sio.loadmat(os.path.join(DATA_DIR, "setid.mat"))

splits = {
    "train": setid["trnid"].squeeze(),
    "val": setid["valid"].squeeze(),
    "test": setid["tstid"].squeeze(),
}

NUM_CLASSES = len(FLOWER_CLASSES)

# ------------------------------------------------------
# TOKENIZER — IDENTICAL TO EVALUATION
# ------------------------------------------------------
def get_tokenizer():
    if MODEL_TYPE.startswith("openai_clip"):
        def tokenizer_fn(prompts, device):
            return clip.tokenize(prompts).to(device), None
        return tokenizer_fn

    elif "bert" in TEXT_ENCODER_NAME:
        tokenizer = BertTokenizer.from_pretrained(TEXT_ENCODER_NAME)
        def tokenizer_fn(prompts, device):
            enc = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            return enc["input_ids"].to(device), enc["attention_mask"].to(device)
        return tokenizer_fn

    elif "gpt" in TEXT_ENCODER_NAME:
        tokenizer = GPT2Tokenizer.from_pretrained(TEXT_ENCODER_NAME)
        tokenizer.pad_token = tokenizer.eos_token
        def tokenizer_fn(prompts, device):
            enc = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            return enc["input_ids"].to(device), enc["attention_mask"].to(device)
        return tokenizer_fn

    else:
        raise ValueError(f"Unknown text encoder: {TEXT_ENCODER_NAME}")

# ------------------------------------------------------
# TEXT FEATURES
# ------------------------------------------------------
prompts = [f"a photo of a {name}" for name in FLOWER_CLASSES]

tokenizer_fn = get_tokenizer()
input_ids, attention_mask = tokenizer_fn(prompts, device)

with torch.no_grad():
    if attention_mask is None:
        text_features = model.encode_text(input_ids)
    else:
        text_features = model.encode_text(input_ids, attention_mask)

    text_features /= text_features.norm(dim=-1, keepdim=True)

# ------------------------------------------------------
# EVALUATION
# ------------------------------------------------------
def evaluate_ids(image_ids, temperature):
    entropies, confidences, logit_stds, gt_probs_all = [], [], [], []

    for i in tqdm(range(0, len(image_ids), BATCH_SIZE), leave=False):
        batch_ids = image_ids[i:i + BATCH_SIZE]
        images, targets = [], []

        for img_id in batch_ids:
            images.append(
                preprocess(Image.open(
                    os.path.join(IMAGE_DIR, f"image_{img_id:05d}.jpg")
                ).convert("RGB"))
            )
            targets.append(labels[img_id - 1])

        images = torch.stack(images).to(device)
        targets = torch.tensor(targets, dtype=torch.long, device=device)

        with torch.no_grad():
            img_feat = model.encode_image(images)
            img_feat /= img_feat.norm(dim=-1, keepdim=True)

            logits = (img_feat @ text_features.T) / temperature
            probs = logits.softmax(dim=-1)

            entropies.extend(
                (-(probs * torch.log(probs + 1e-8)).sum(dim=-1)).cpu().tolist()
            )
            confidences.extend(probs.max(dim=-1).values.cpu().tolist())
            logit_stds.extend(logits.std(dim=-1).cpu().tolist())

            gt_probs_all.append(
                probs.gather(1, targets.unsqueeze(1)).squeeze(1).cpu()
            )

    return (
        np.mean(entropies),
        np.mean(confidences),
        np.mean(logit_stds),
        torch.cat(gt_probs_all),
    )

# ------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------
summary_rows = []
mean_conf_vs_T = []
mean_entropy_vs_T = []

for T in TEMPERATURES:
    print(f"\n===== TEMPERATURE {T} =====")

    all_gt_probs = []
    entropies_T = []
    confidences_T = []

    for split_name, split_ids in splits.items():
        ent, conf, lstd, gt_probs = evaluate_ids(split_ids, T)

        entropies_T.append(ent)
        confidences_T.append(conf)
        all_gt_probs.append(gt_probs)

        summary_rows.extend([
            {"model": MODEL_ID, "temperature": T, "split": split_name,
             "metric": "entropy", "mean": ent},
            {"model": MODEL_ID, "temperature": T, "split": split_name,
             "metric": "confidence", "mean": conf},
            {"model": MODEL_ID, "temperature": T, "split": split_name,
             "metric": "logit_std", "mean": lstd},
        ])

    mean_entropy_vs_T.append(np.mean(entropies_T))
    mean_conf_vs_T.append(np.mean(confidences_T))

    # Ground-truth probability histogram
    probs = torch.cat(all_gt_probs).numpy()
    plt.figure(figsize=(7, 5))
    plt.hist(probs, bins=50, density=True)
    plt.xlabel("P(correct class)")
    plt.ylabel("Density")
    plt.title(f"{MODEL_ID} | Ground-truth Probability (T={T})")
    plt.tight_layout()
    plt.savefig(os.path.join(
        RESULTS_DIR, f"gt_probability_histogram_T{T}.png"
    ))
    plt.close()

# ------------------------------------------------------
# ENTROPY VS TEMPERATURE (RESTORED)
# ------------------------------------------------------
plt.figure(figsize=(7, 5))
plt.plot(TEMPERATURES, mean_entropy_vs_T, marker="o")
plt.xlabel("Temperature τ")
plt.ylabel("Mean entropy")
plt.title(f"Entropy vs Temperature ({MODEL_ID})")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "entropy_vs_temperature.png"))
plt.close()

# ------------------------------------------------------
# MEAN CONFIDENCE VS TEMPERATURE
# ------------------------------------------------------
plt.figure(figsize=(7, 5))
plt.plot(TEMPERATURES, mean_conf_vs_T, marker="o")
plt.xlabel("Temperature τ")
plt.ylabel("Mean max probability")
plt.title(f"Mean Confidence vs Temperature ({MODEL_ID})")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "mean_confidence_vs_temperature.png"))
plt.close()

# ------------------------------------------------------
# SAVE CSV
# ------------------------------------------------------
df = pd.DataFrame(summary_rows)
df.to_csv(os.path.join(RESULTS_DIR, "summary_metrics.csv"), index=False)

print(f"\nAll temperature results saved to {RESULTS_DIR}")
