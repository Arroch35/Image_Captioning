#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import clip
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

# ------------------------------------------------------
# CONFIG
# ------------------------------------------------------
DATA_DIR = "../data"
IMAGE_DIR = os.path.join(DATA_DIR, "jpg")
RESULTS_DIR = "../data/results"
BATCH_SIZE = 32
N_SPLITS = 10

os.makedirs(RESULTS_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ------------------------------------------------------
# LOAD CLIP
# ------------------------------------------------------
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# ------------------------------------------------------
# FLOWER CLASS NAMES (102)
# ------------------------------------------------------
FLOWER_CLASSES = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells",
    "sweet pea", "english marigold", "tiger lily", "moon orchid",
    "bird of paradise", "monkshood", "globe thistle",
    "snapdragon", "colt's foot", "king protea", "spear thistle",
    "yellow iris", "globe-flower", "purple coneflower", "peruvian lily",
    "balloon flower", "giant white arum lily", "fire lily", "pincushion flower",
    "fritillary", "red ginger", "grape hyacinth", "corn poppy",
    "prince of wales feathers", "stemless gentian", "artichoke",
    "sweet william", "carnation", "garden phlox", "love in the mist",
    "mexican aster", "alpine sea holly", "ruby-lipped cattleya",
    "cape flower", "great masterwort", "siam tulip", "lenten rose",
    "barbeton daisy", "daffodil", "sword lily", "poinsettia",
    "bolero deep blue", "wallflower", "marigold", "buttercup",
    "oxeye daisy", "common dandelion", "petunia", "wild pansy",
    "primula", "sunflower", "pelargonium", "bishop of llandaff",
    "gaura", "geranium", "orange dahlia", "pink-yellow dahlia",
    "cautleya spicata", "japanese anemone", "black-eyed susan",
    "silverbush", "californian poppy", "osteospermum",
    "spring crocus", "bearded iris", "windflower", "tree poppy",
    "gazania", "azalea", "water lily", "rose", "thorn apple",
    "morning glory", "passion flower", "lotus lotus",
    "toad lily", "anthurium", "frangipani", "clematis",
    "hibiscus", "columbine", "desert rose", "tree mallow",
    "magnolia", "cyclamen", "watercress", "canna lily",
    "hippeastrum", "bee balm", "ball moss", "foxglove",
    "bougainvillea", "camellia", "mallow", "mexican petunia",
    "bromelia", "blanket flower", "trumpet creeper",
    "blackberry lily"
]

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
# PREPARE TEXT FEATURES (ONCE)
# ------------------------------------------------------
prompts = [f"a photo of a {name}" for name in FLOWER_CLASSES]
text_tokens = clip.tokenize(prompts).to(device)

with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

logit_scale = model.logit_scale.exp()

# ------------------------------------------------------
# EVALUATION FUNCTION
# ------------------------------------------------------
def evaluate_ids(image_ids):
    top1 = top3 = top5 = 0
    similarity_sum = 0.0
    all_logits = []
    all_targets = []

    for i in tqdm(range(0, len(image_ids), BATCH_SIZE), leave=False):
        batch_ids = image_ids[i:i + BATCH_SIZE]
        images = []
        targets = []

        for img_id in batch_ids:
            path = os.path.join(IMAGE_DIR, f"image_{img_id:05d}.jpg")
            images.append(preprocess(Image.open(path)))
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
# K-FOLD CV PER SPLIT
# ------------------------------------------------------
all_results = {}

for split_name, split_ids in splits.items():
    print(f"\n===== {split_name.upper()} SPLIT =====")

    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    split_metrics = {
        "top1": [],
        "top3": [],
        "top5": [],
        "mean_similarity": [],
    }

    roc_logits = []
    roc_targets = []
    
    split_labels = labels[split_ids - 1]

    for fold, (_, test_idx) in enumerate(kf.split(split_ids, split_labels)):
        print(f"  Fold {fold + 1}/{N_SPLITS}")
        fold_ids = split_ids[test_idx]

        t1, t3, t5, ms, logits, targets = evaluate_ids(fold_ids)

        split_metrics["top1"].append(t1)
        split_metrics["top3"].append(t3)
        split_metrics["top5"].append(t5)
        split_metrics["mean_similarity"].append(ms)

        roc_logits.append(logits)
        roc_targets.append(targets)

    all_results[split_name] = {
        "metrics": split_metrics,
        "roc_logits": roc_logits,
        "roc_targets": roc_targets,
    }

# ------------------------------------------------------
# BOXPLOTS PER SPLIT
# ------------------------------------------------------
for split_name, result in all_results.items():
    metrics = result["metrics"]

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=[
        metrics["top1"],
        metrics["top3"],
        metrics["top5"],
        metrics["mean_similarity"],
    ])

    plt.xticks([0, 1, 2, 3],
               ["Top-1", "Top-3", "Top-5", "Mean Similarity"])
    plt.ylabel("Score")
    plt.title(f"CLIP Zero-Shot ({split_name} split, {N_SPLITS}-fold)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{split_name}_metrics_boxplot.png"))
    plt.close()

# ------------------------------------------------------
# ROC CURVES PER SPLIT
# ------------------------------------------------------
for split_name, result in all_results.items():
    y_true = []
    y_score = []

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
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve – {split_name} split")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{split_name}_roc_curve.png"))
    plt.close()

# ------------------------------------------------------
# FINAL SUMMARY (MEAN ± STD)
# ------------------------------------------------------

summary_rows = []

print("\n===== FINAL RESULTS (mean ± std over k folds) =====")

for split_name, result in all_results.items():
    print(f"\n{split_name.upper()} SPLIT:")
    for metric, values in result["metrics"].items():
        mean = np.mean(values)
        std = np.std(values)
        print(f"  {metric.replace('_', ' ').title():18s}: {mean:.4f} ± {std:.4f}")
        
        summary_rows.append({
            "split": split_name,
            "metric": metric,
            "mean": mean,
            "std": std,
        })

summary_df = pd.DataFrame(summary_rows)

csv_path = os.path.join(RESULTS_DIR, "summary_metrics.csv")
summary_df.round(4).to_csv(csv_path, index=False)

print(f"\nResults saved to: {csv_path}")