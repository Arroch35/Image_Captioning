#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
from PIL import Image
import scipy.io as sio
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import clip
from transformers import BertTokenizer
from flowers_names import FLOWER_CLASSES
from utils import build_clip_preprocess, build_custom_resnet50_bert_clip

# ------------------------------------------------------
# CONFIG
# ------------------------------------------------------
MODEL_NAME = "ViT-B-32_zero_shot" # "resnet50_bert" #"ViT-B-32_zero_shot"          # used for filenames
MODEL_TYPE = "openai_clip"   # openai_clip | openai_clip_finetuned | custom_clip
CLIP_BACKBONE = "ViT-B/32"   # only for openai_clip and openai_clip_finetuned
CHECKPOINT_PATH = "../models/custom_clip_resnet50_bert.pth"   # path/to/model.pth for custom

PART = "/part1" # Change depending on the part of the project
DATA_DIR = "../data"
IMAGE_DIR = os.path.join(DATA_DIR, "jpg")
RESULTS_DIR = "../data/results" + PART
TEMP_RESULTS_DIR = os.path.join(RESULTS_DIR, "temperature", MODEL_NAME)

BATCH_SIZE = 32
TEMPERATURES = [0.005, 0.01, 0.02, 0.05, 0.1]

os.makedirs(TEMP_RESULTS_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print(f"Model: {MODEL_NAME}")

# ------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------
def load_model():
    if MODEL_TYPE == "openai_clip":
        model, preprocess = clip.load(CLIP_BACKBONE, device=device)
        model.eval()
        return model, preprocess

    elif MODEL_TYPE == "openai_clip_finetuned":
        model, preprocess = clip.load(CLIP_BACKBONE, device=device)
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(ckpt)
        model.eval()
        return model, preprocess

    elif MODEL_TYPE == "custom_clip":
        model = build_custom_resnet50_bert_clip(device)
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
        model.eval()
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
# TEXT FEATURES
# ------------------------------------------------------
prompts = [f"a photo of a {name}" for name in FLOWER_CLASSES]

def tokenize(prompts):
    if MODEL_TYPE.startswith("openai_clip"):
        return clip.tokenize(prompts).to(device), None
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    enc = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)

input_ids, attention_mask = tokenize(prompts)

with torch.no_grad():
    if attention_mask is None:
        text_features = model.encode_text(input_ids)
    else:
        text_features = model.encode_text(input_ids, attention_mask)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# ------------------------------------------------------
# EVALUATION
# ------------------------------------------------------
def evaluate_ids(image_ids, temperature, collect_probs=False):
    entropies, confidences, logit_stds = [], [], []
    all_probs = []

    for i in tqdm(range(0, len(image_ids), BATCH_SIZE), leave=False):
        batch_ids = image_ids[i:i + BATCH_SIZE]
        images = []

        for img_id in batch_ids:
            path = os.path.join(IMAGE_DIR, f"image_{img_id:05d}.jpg")
            images.append(preprocess(Image.open(path).convert("RGB")))

        images = torch.stack(images).to(device)

        with torch.no_grad():
            img_feat = model.encode_image(images)
            img_feat /= img_feat.norm(dim=-1, keepdim=True)

            logits = (img_feat @ text_features.T) / temperature
            probs = logits.softmax(dim=-1)

            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
            confidence = probs.max(dim=-1).values
            logit_std = logits.std(dim=-1)

            entropies.extend(entropy.cpu().tolist())
            confidences.extend(confidence.cpu().tolist())
            logit_stds.extend(logit_std.cpu().tolist())

            if collect_probs:
                all_probs.append(probs.flatten().cpu())

    if collect_probs:
        return (
            np.mean(entropies),
            np.mean(confidences),
            np.mean(logit_stds),
            torch.cat(all_probs),
        )

    return np.mean(entropies), np.mean(confidences), np.mean(logit_stds)

# ------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------
summary_rows = []
entropy_vs_T = []

for T in TEMPERATURES:
    print(f"\n===== TEMPERATURE {T} =====")
    temp_probs = []

    for split_name, split_ids in splits.items():
        ent, conf, lstd, probs = evaluate_ids(split_ids, T, collect_probs=True)
        temp_probs.append(probs)

        summary_rows.extend([
            {"model": MODEL_NAME, "temperature": T, "split": split_name,
             "metric": "entropy", "mean": ent},
            {"model": MODEL_NAME, "temperature": T, "split": split_name,
             "metric": "confidence", "mean": conf},
            {"model": MODEL_NAME, "temperature": T, "split": split_name,
             "metric": "logit_std", "mean": lstd},
        ])

    entropy_vs_T.append(ent)

    # Dataset-level probability histogram
    all_probs = torch.cat(temp_probs).numpy()
    plt.figure(figsize=(7, 5))
    plt.hist(all_probs, bins=50, density=True)
    plt.xlabel("Predicted probability")
    plt.ylabel("Density")
    plt.title(f"Probability Distribution | T={T}")
    plt.tight_layout()
    plt.savefig(os.path.join(TEMP_RESULTS_DIR, f"probability_histogram_T{T}.png"))
    plt.close()

# ------------------------------------------------------
# ENTROPY VS TEMPERATURE PLOT
# ------------------------------------------------------
plt.figure(figsize=(7, 5))
plt.plot(TEMPERATURES, entropy_vs_T, marker="o")
plt.xlabel("Temperature τ")
plt.ylabel("Mean entropy")
plt.title(f"Entropy vs Temperature ({MODEL_NAME})")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(TEMP_RESULTS_DIR, "entropy_vs_temperature.png"))
plt.close()

# ------------------------------------------------------
# SAVE CSV
# ------------------------------------------------------
df = pd.DataFrame(summary_rows)
csv_path = os.path.join(TEMP_RESULTS_DIR, "summary_metrics.csv")
df.round(4).to_csv(csv_path, index=False)

print(f"\nAll temperature results saved to {TEMP_RESULTS_DIR}")
