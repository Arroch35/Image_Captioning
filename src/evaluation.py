#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
from PIL import Image
import scipy.io as sio
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix

import clip
from transformers import BertTokenizer, GPT2Tokenizer

from flowers_names import FLOWER_CLASSES
from utils import *
from P3_Models import *

# ------------------------------------------------------
# CONFIG (CHANGE ONLY THIS)
# ------------------------------------------------------
MODEL_TYPE = "openai_clip"
# "openai_clip" | "openai_clip_finetuned" | "custom_clip"

# --- OpenAI CLIP ---
CLIP_BACKBONE = "ViT-B/32"

# --- Custom CLIP ---
IMAGE_ENCODER_NAME = "swin_tiny" # resnet50 | swin_tiny
TEXT_ENCODER_NAME  = "gpt2" # bert-base-uncased | gpt2
CHECKPOINT_PATH = f"../models/cosine_lr/custom_clip_{IMAGE_ENCODER_NAME}_{TEXT_ENCODER_NAME}.pth"

# MODEL ID (used for result folder naming)
if MODEL_TYPE == "custom_clip":
    MODEL_ID = f"{IMAGE_ENCODER_NAME}_{TEXT_ENCODER_NAME}"
else:
    MODEL_ID = CLIP_BACKBONE.replace("/", "_")

PART = "/part1" # Change for each part of the project
DATA_DIR = "../data"
IMAGE_DIR = os.path.join(DATA_DIR, "jpg")
BASE_RESULTS_DIR = "../data/results" + PART
RESULTS_DIR = os.path.join(BASE_RESULTS_DIR, MODEL_ID)

os.makedirs(RESULTS_DIR, exist_ok=True)

BATCH_SIZE = 64
N_SPLITS = 10
EMBED_DIM = 512
TEMPERATURES = {
    ("resnet50", "bert-base-uncased"): 0.0688,
    ("swin_tiny", "bert-base-uncased"): 0.0675,
    ("resnet50", "gpt2"): 0.0678,
    ("swin_tiny", "gpt2"): 0.0674,
}

LEARNED_TEMPERATURE = TEMPERATURES[(IMAGE_ENCODER_NAME, TEXT_ENCODER_NAME)]

os.makedirs(RESULTS_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print(f"Model type: {MODEL_TYPE}")

# ------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------
def load_model():
    if MODEL_TYPE == "openai_clip":
        model, preprocess = clip.load(CLIP_BACKBONE, device=device)
        model.eval()
        return model, preprocess, "openai"

    elif MODEL_TYPE == "openai_clip_finetuned":
        model, preprocess = clip.load(CLIP_BACKBONE, device=device)
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device).eval()
        return model, preprocess, "openai"

    elif MODEL_TYPE == "custom_clip":
        image_encoder = build_image_encoder(IMAGE_ENCODER_NAME, EMBED_DIM)
        text_encoder  = build_text_encoder(TEXT_ENCODER_NAME, EMBED_DIM)

        model = ModularCLIP(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            embed_dim=EMBED_DIM,
            learned_temperature=LEARNED_TEMPERATURE
        )

        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device).eval()

        preprocess = build_clip_preprocess(image_size=224)
        return model, preprocess, "custom"

    else:
        raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")

model, preprocess, mode = load_model()

# ------------------------------------------------------
# LOAD LABELS & SPLITS
# ------------------------------------------------------
labels = sio.loadmat(os.path.join(DATA_DIR, "imagelabels.mat"))["labels"].squeeze() - 1
setid = sio.loadmat(os.path.join(DATA_DIR, "setid.mat"))

trnid = setid["trnid"].squeeze()
valid = setid["valid"].squeeze()
tstid = setid["tstid"].squeeze()

splits = {
    "train": np.concatenate([valid, tstid]),  # <-- evaluate here
    "reversed_test": trnid,   # <-- evaluate here only
}

NUM_CLASSES = len(FLOWER_CLASSES)

# ------------------------------------------------------
# TEXT FEATURES (ONCE)
# ------------------------------------------------------
def get_tokenizer(name):
    if "bert" in name:
        return BertTokenizer.from_pretrained(name)
    elif "gpt" in name:
        tok = GPT2Tokenizer.from_pretrained(name)
        tok.pad_token = tok.eos_token
        return tok
    else:
        raise ValueError(f"Unknown text encoder: {name}")

prompts = [f"a photo of a {name}" for name in FLOWER_CLASSES]

with torch.no_grad():
    if mode == "openai":
        tokens = clip.tokenize(prompts).to(device)
        text_features = model.encode_text(tokens)

    else:  # custom clip
        tokenizer = get_tokenizer(TEXT_ENCODER_NAME)
        encoded = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        text_features = model.encode_text(input_ids, attention_mask)

text_features = text_features / text_features.norm(dim=-1, keepdim=True)
logit_scale = model.logit_scale.exp()

# ------------------------------------------------------
# EVALUATION FUNCTION
# ------------------------------------------------------
def evaluate_ids(image_ids):
    top1 = top3 = top5 = 0
    clip_similarity_sum = 0.0      # CLIP-style similarity (with learned temperature)
    cosine_similarity_sum = 0.0    # pure cosine similarity (0..1)
    all_logits, all_targets = [], []
    top1_preds= []

    for i in tqdm(range(0, len(image_ids), BATCH_SIZE), leave=False):
        batch_ids = image_ids[i:i + BATCH_SIZE]
        images, targets = [], []

        for img_id in batch_ids:
            path = os.path.join(IMAGE_DIR, f"image_{img_id:05d}.jpg")
            images.append(preprocess(Image.open(path).convert("RGB")))
            targets.append(labels[img_id - 1])

        images = torch.stack(images).to(device)
        targets = torch.tensor(targets, dtype=torch.long, device=device)

        with torch.no_grad():
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            logits = logit_scale * image_features @ text_features.T
            _, topk = logits.topk(5, dim=-1)

            gt_text = text_features.index_select(0, targets)
            cosine_similarity_sum += ((image_features * gt_text).sum(dim=-1)).sum().item()
            clip_similarity_sum += (logit_scale * (image_features * gt_text).sum(dim=-1)).sum().item()

        top1 += (topk[:, 0] == targets).sum().item()
        top3 += sum(t in topk[i, :3] for i, t in enumerate(targets))
        top5 += sum(t in topk[i] for i, t in enumerate(targets))

        top1_preds.extend(topk[:, 0].cpu().numpy())
        all_logits.append(logits.cpu())
        all_targets.append(targets.cpu())

    total = len(image_ids)
    return (
        top1 / total,
        top3 / total,
        top5 / total,
        cosine_similarity_sum / total,
        clip_similarity_sum / total,
        torch.cat(all_logits),
        torch.cat(all_targets),
        np.array(top1_preds),
    )

# ------------------------------------------------------
# K-FOLD EVALUATION
# ------------------------------------------------------
all_results = {}

for split_name, split_ids in splits.items():
    print(f"\n===== {split_name.upper()} SPLIT =====")

    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    split_labels = labels[split_ids - 1]

    metrics = {"top1": [], "top3": [], "top5": [], "mean_cosine_similarity": [], "mean_clip_similarity": []}
    roc_logits, roc_targets = [], []
    all_fold_targets = []
    all_fold_preds = []     

    for fold, (_, test_idx) in enumerate(kf.split(split_ids, split_labels)):
        print(f"  Fold {fold + 1}/{N_SPLITS}")
        fold_ids = split_ids[test_idx]

        t1, t3, t5, coss, clis, logits, targets, preds = evaluate_ids(fold_ids)
        metrics["top1"].append(t1)
        metrics["top3"].append(t3)
        metrics["top5"].append(t5)
        metrics["mean_cosine_similarity"].append(coss)
        metrics["mean_clip_similarity"].append(clis)

        roc_logits.append(logits)
        roc_targets.append(targets)
        
        all_fold_targets.append(targets.numpy())
        all_fold_preds.append(preds)

    
    all_targets_concat = np.concatenate(all_fold_targets)
    all_preds_concat = np.concatenate(all_fold_preds)
    
    cm = confusion_matrix(all_targets_concat, all_preds_concat, labels=np.arange(NUM_CLASSES))
    df_cm = pd.DataFrame(cm, index=FLOWER_CLASSES, columns=FLOWER_CLASSES)
    cm_csv_path = os.path.join(RESULTS_DIR, f"{MODEL_TYPE}_{split_name}_confusion_matrix.csv")
    df_cm.to_csv(cm_csv_path)
    print(f"Confusion matrix for {split_name} saved to {cm_csv_path}")
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(df_cm, annot=False, fmt="d", cmap="Blues")  # annot=True if you want numbers
    plt.title(f"{MODEL_TYPE} – {split_name} Confusion Matrix")
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{MODEL_TYPE}_{split_name}_confusion_matrix.png"))
    plt.close()

    all_results[split_name] = {
        "metrics": metrics,
        "roc_logits": roc_logits,
        "roc_targets": roc_targets,
    }

# ------------------------------------------------------
# BOXPLOTS
# ------------------------------------------------------
for split_name, result in all_results.items():
    m = result["metrics"]
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=[m["top1"], m["top3"], m["top5"], m["mean_clip_similarity"]])
    plt.xticks([0, 1, 2, 3], ["Top-1", "Top-3", "Top-5", "Mean Similarity"])
    plt.title(f"{MODEL_TYPE} – {split_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{MODEL_TYPE}_{split_name}_clip_similarity_boxplot.png"))
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=[m["top1"], m["top3"], m["top5"], m["mean_cosine_similarity"]])
    plt.xticks([0, 1, 2, 3], ["Top-1", "Top-3", "Top-5", "Mean Similarity"])
    plt.title(f"{MODEL_TYPE} – {split_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{MODEL_TYPE}_{split_name}_cosine_similarity_boxplot.png"))
    plt.close()

# ------------------------------------------------------
# ROC CURVES
# ------------------------------------------------------
for split_name, result in all_results.items():
    y_true, y_score = [], []

    for logits, targets in zip(result["roc_logits"], result["roc_targets"]):
        y_true.append(label_binarize(targets.numpy(), classes=np.arange(NUM_CLASSES)))
        y_score.append(logits.softmax(dim=-1).numpy())

    y_true = np.vstack(y_true)
    y_score = np.vstack(y_score)

    fpr, tpr, _ = roc_curve(y_true.ravel(), y_score.ravel())
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.title(f"ROC – {MODEL_TYPE} – {split_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{MODEL_TYPE}_{split_name}_roc.png"))
    plt.close()

# ------------------------------------------------------
# SUMMARY CSV
# ------------------------------------------------------
rows = []
for split_name, result in all_results.items():
    for metric, values in result["metrics"].items():
        rows.append({
            "model": MODEL_TYPE,
            "split": split_name,
            "metric": metric,
            "mean": np.mean(values),
            "std": np.std(values),
        })

summary_df = pd.DataFrame(rows)
csv_path = os.path.join(RESULTS_DIR, f"{MODEL_TYPE}_summary_metrics.csv")
summary_df.round(4).to_csv(csv_path, index=False)

print(f"\nResults saved to {csv_path}")
