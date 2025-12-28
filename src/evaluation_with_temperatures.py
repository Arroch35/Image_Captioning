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

import clip
from transformers import BertTokenizer
from flowers_names import FLOWER_CLASSES
from utils import build_clip_preprocess, build_custom_resnet50_bert_clip

# ------------------------------------------------------
# CONFIG - CHANGE MODEL HERE
# ------------------------------------------------------
MODEL_NAME = "resnet50_bert"          # used in filenames
MODEL_TYPE = "custom_clip"            # openai_clip | openai_clip_finetuned | custom_clip
CLIP_BACKBONE = "ViT-B/32"             # for OpenAI CLIP
CHECKPOINT_PATH = "../models/custom_clip_resnet50_bert.pth"

PART = "/part3"
DATA_DIR = "../data"
IMAGE_DIR = os.path.join(DATA_DIR, "jpg")

BASE_RESULTS_DIR = "../data/results" + PART
TEMP_RESULTS_DIR = os.path.join(BASE_RESULTS_DIR, "temperature", MODEL_NAME)

BATCH_SIZE = 32
N_SPLITS = 10

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
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint)
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
# TOKENIZER + TEXT FEATURES
# ------------------------------------------------------
prompts = [f"a photo of a {name}" for name in FLOWER_CLASSES]

def tokenize_text(prompts):
    if MODEL_TYPE.startswith("openai_clip"):
        tokens = clip.tokenize(prompts).to(device)
        return tokens, None
    else:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        enc = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
        return enc["input_ids"].to(device), enc["attention_mask"].to(device)

input_ids, attention_mask = tokenize_text(prompts)

with torch.no_grad():
    if attention_mask is None:
        text_features = model.encode_text(input_ids)
    else:
        text_features = model.encode_text(input_ids, attention_mask)

    text_features /= text_features.norm(dim=-1, keepdim=True)

# ------------------------------------------------------
# EVALUATION FUNCTION
# ------------------------------------------------------
def evaluate_ids(image_ids, temperature):
    top1 = top3 = top5 = 0
    similarity_sum = 0.0
    all_logits, all_targets = [], []

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

            logits = (image_features @ text_features.T) / temperature
            _, topk = logits.topk(5, dim=-1)

            gt = text_features.index_select(0, targets)
            similarity_sum += (image_features * gt).sum(dim=-1).sum().item()

        top1 += (topk[:, 0] == targets).sum().item()
        top3 += sum(t in topk[i, :3] for i, t in enumerate(targets))
        top5 += sum(t in topk[i] for i, t in enumerate(targets))

        all_logits.append(logits.cpu())
        all_targets.append(targets.cpu())

    total = len(image_ids)
    return (
        top1 / total,
        top3 / total,
        top5 / total,
        similarity_sum / total,
        torch.cat(all_logits),
        torch.cat(all_targets),
    )

# ------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------
summary_rows = []

for T in TEMPERATURES:
    print(f"\n===== TEMPERATURE {T} =====")

    for split_name, split_ids in splits.items():
        print(f"--- {split_name.upper()} ---")

        kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
        split_labels = labels[split_ids - 1]

        metrics = {"top1": [], "top3": [], "top5": [], "mean_similarity": []}
        roc_logits, roc_targets = [], []

        for _, test_idx in kf.split(split_ids, split_labels):
            fold_ids = split_ids[test_idx]
            t1, t3, t5, ms, logits, targets = evaluate_ids(fold_ids, T)

            metrics["top1"].append(t1)
            metrics["top3"].append(t3)
            metrics["top5"].append(t5)
            metrics["mean_similarity"].append(ms)

            roc_logits.append(logits)
            roc_targets.append(targets)

        # Save plots
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=list(metrics.values()))
        plt.xticks(range(4), ["Top-1", "Top-3", "Top-5", "Mean Similarity"])
        plt.title(f"{MODEL_NAME} | {split_name} | T={T}")
        plt.tight_layout()
        plt.savefig(os.path.join(
            TEMP_RESULTS_DIR, f"{split_name}_T{T}_boxplot.png"
        ))
        plt.close()

        # ROC
        y_true = np.vstack([
            label_binarize(t.numpy(), classes=np.arange(NUM_CLASSES))
            for t in roc_targets
        ])
        y_score = np.vstack([l.softmax(dim=-1).numpy() for l in roc_logits])

        fpr, tpr, _ = roc_curve(y_true.ravel(), y_score.ravel())
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(
            TEMP_RESULTS_DIR, f"{split_name}_T{T}_roc.png"
        ))
        plt.close()

        for m, v in metrics.items():
            summary_rows.append({
                "model": MODEL_NAME,
                "temperature": T,
                "split": split_name,
                "metric": m,
                "mean": np.mean(v),
                "std": np.std(v),
            })

# ------------------------------------------------------
# SAVE CSV
# ------------------------------------------------------
df = pd.DataFrame(summary_rows)
csv_path = os.path.join(TEMP_RESULTS_DIR, "summary_metrics.csv")
df.round(4).to_csv(csv_path, index=False)

print(f"\nAll temperature results saved to {TEMP_RESULTS_DIR}")
