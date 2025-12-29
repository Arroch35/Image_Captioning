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

import clip  # OpenAI CLIP
from flowers_names import FLOWER_CLASSES
from utils import build_clip_preprocess, build_custom_resnet50_bert_clip
from transformers import BertModel, BertTokenizer


# ------------------------------------------------------
# CONFIG — CHANGE MODEL HERE
# ------------------------------------------------------
MODEL_NAME = "resnet50_bert" #"resnet50_bert" #"ViT-B-32_zero_shot"          # used for filenames
MODEL_TYPE = "custom_clip"                 # "openai_clip" | "openai_clip_finetuned" | "custom_clip"
CLIP_BACKBONE = "ViT-B/32"                 # only for openai_clip and openai_clip_finetuned
CHECKPOINT_PATH = "../models/custom_clip_resnet50_bert.pth"   # path/to/model.pth for custom

PART = "/part3" # Change depending on the part of the project
DATA_DIR = "../data"
IMAGE_DIR = os.path.join(DATA_DIR, "jpg")
RESULTS_DIR = "../data/results" + PART
BATCH_SIZE = 32
N_SPLITS = 10

os.makedirs(RESULTS_DIR, exist_ok=True)

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
        # Fine-tuned OpenAI CLIP
        model, preprocess = clip.load(CLIP_BACKBONE, device=device)

        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

        model.load_state_dict(checkpoint)

        model.to(device).eval()
        return model, preprocess

    elif MODEL_TYPE == "custom_clip":
        from P3_Models import ModularCLIP  # your model file
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model = build_custom_resnet50_bert_clip(device)
        
        model.load_state_dict(checkpoint)
        model.to(device).eval()

        preprocess = build_clip_preprocess(image_size=224)
        return model, preprocess

    else:
        raise ValueError("Unknown MODEL_TYPE")

model, preprocess = load_model()

# ------------------------------------------------------
# FLOWER CLASS NAMES
# ------------------------------------------------------
NUM_CLASSES = len(FLOWER_CLASSES)

# ------------------------------------------------------
# LOAD LABELS & SPLITS
# ------------------------------------------------------
labels = sio.loadmat(os.path.join(DATA_DIR, "imagelabels.mat"))["labels"].squeeze() - 1
setid = sio.loadmat(os.path.join(DATA_DIR, "setid.mat"))

splits = {
    "train": setid["trnid"].squeeze(),
    "val": setid["valid"].squeeze(),
    "test": setid["tstid"].squeeze(),
}

# ------------------------------------------------------
# TEXT FEATURES (ONCE)
# ------------------------------------------------------
prompts = [f"a photo of a {name}" for name in FLOWER_CLASSES]

def get_tokenizer(model_name_or_type):
    if model_name_or_type.lower().startswith("openai_clip"):
        def tokenizer(prompts, device):
            tokens = clip.tokenize(prompts).to(device)
            return tokens, None
        return tokenizer

    elif model_name_or_type.lower().endswith("bert"):
        bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        def tokenizer(prompts, device):
            encoded = bert_tokenizer(
                prompts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            return (
                encoded["input_ids"].to(device),
                encoded["attention_mask"].to(device)
            )
        return tokenizer

    else:
        raise ValueError(f"Unknown model type: {model_name_or_type}")

# 1 Get tokenizer function
tokenizer_fn = get_tokenizer(
    MODEL_NAME if MODEL_TYPE == "custom_clip" else MODEL_TYPE
)

# 2 Run tokenizer
text_tokens, attention_mask = tokenizer_fn(prompts, device)

# 3 Encode text
if MODEL_TYPE == "custom_clip" and MODEL_NAME.lower().endswith("bert"):
    with torch.no_grad():
        text_features = model.encode_text(text_tokens, attention_mask)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
else:
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)


logit_scale = model.logit_scale.exp()

# ------------------------------------------------------
# EVALUATION
# ------------------------------------------------------
def evaluate_ids(image_ids):
    top1 = top3 = top5 = 0
    similarity_sum = 0.0
    all_logits = []
    all_targets = []

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

            gt_text_features = text_features.index_select(0, targets)
            similarity_sum += (image_features * gt_text_features).sum(dim=-1).sum().item()

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
# K-FOLD EVALUATION
# ------------------------------------------------------
all_results = {}

for split_name, split_ids in splits.items():
    print(f"\n===== {split_name.upper()} SPLIT =====")

    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    split_labels = labels[split_ids - 1]

    metrics = {"top1": [], "top3": [], "top5": [], "mean_similarity": []}
    roc_logits, roc_targets = [], []

    for fold, (_, test_idx) in enumerate(kf.split(split_ids, split_labels)):
        print(f"  Fold {fold + 1}/{N_SPLITS}")
        fold_ids = split_ids[test_idx]

        t1, t3, t5, ms, logits, targets = evaluate_ids(fold_ids)
        metrics["top1"].append(t1)
        metrics["top3"].append(t3)
        metrics["top5"].append(t5)
        metrics["mean_similarity"].append(ms)

        roc_logits.append(logits)
        roc_targets.append(targets)

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
    sns.boxplot(data=[m["top1"], m["top3"], m["top5"], m["mean_similarity"]])
    plt.xticks([0, 1, 2, 3], ["Top-1", "Top-3", "Top-5", "Mean Similarity"])
    plt.title(f"{MODEL_NAME} – {split_name} split")
    plt.tight_layout()

    plt.savefig(os.path.join(
        RESULTS_DIR, f"{MODEL_NAME}_{split_name}_metrics_boxplot.png"
    ))
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
    plt.title(f"ROC – {MODEL_NAME} – {split_name}")
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(
        RESULTS_DIR, f"{MODEL_NAME}_{split_name}_roc_curve.png"
    ))
    plt.close()

# ------------------------------------------------------
# SUMMARY CSV
# ------------------------------------------------------
rows = []

for split_name, result in all_results.items():
    for metric, values in result["metrics"].items():
        rows.append({
            "model": MODEL_NAME,
            "split": split_name,
            "metric": metric,
            "mean": np.mean(values),
            "std": np.std(values),
        })

summary_df = pd.DataFrame(rows)
csv_path = os.path.join(RESULTS_DIR, f"{MODEL_NAME}_summary_metrics.csv")
summary_df.round(4).to_csv(csv_path, index=False)

print(f"\nResults saved to {csv_path}")
