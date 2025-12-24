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

# Temperature sweep
TEMPERATURES = [0.005, 0.01, 0.02, 0.05, 0.1]

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

# ------------------------------------------------------
# EVALUATION FUNCTION
# ------------------------------------------------------
def evaluate_ids(image_ids, temperature):
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

            logits = (image_features @ text_features.T) / temperature
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
# MAIN EXPERIMENT LOOP
# ------------------------------------------------------
summary_rows = []

for temperature in TEMPERATURES:
    print(f"\n==============================")
    print(f" TEMPERATURE = {temperature}")
    print(f"==============================")

    for split_name, split_ids in splits.items():
        print(f"\n--- {split_name.upper()} SPLIT ---")

        split_labels = labels[split_ids - 1]
        kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

        metrics = {"top1": [], "top3": [], "top5": [], "mean_similarity": []}
        roc_logits, roc_targets = [], []

        for fold, (_, test_idx) in enumerate(kf.split(split_ids, split_labels)):
            print(f"Fold {fold + 1}/{N_SPLITS}")

            fold_ids = split_ids[test_idx]
            t1, t3, t5, ms, logits, targets = evaluate_ids(fold_ids, temperature)

            metrics["top1"].append(t1)
            metrics["top3"].append(t3)
            metrics["top5"].append(t5)
            metrics["mean_similarity"].append(ms)

            roc_logits.append(logits)
            roc_targets.append(targets)

        # Save summary
        for metric, values in metrics.items():
            summary_rows.append({
                "temperature": temperature,
                "split": split_name,
                "metric": metric,
                "mean": np.mean(values),
                "std": np.std(values),
            })

        # Boxplot
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=list(metrics.values()))
        plt.xticks(range(4), ["Top-1", "Top-3", "Top-5", "Mean Similarity"])
        plt.title(f"{split_name} | T={temperature}")
        plt.tight_layout()
        plt.savefig(os.path.join(
            RESULTS_DIR, f"{split_name}_T{temperature}_boxplot.png"))
        plt.close()

        # ROC
        y_true, y_score = [], []
        for lg, tg in zip(roc_logits, roc_targets):
            y_true.append(label_binarize(tg.numpy(), classes=np.arange(NUM_CLASSES)))
            y_score.append(lg.softmax(dim=-1).numpy())

        y_true = np.vstack(y_true)
        y_score = np.vstack(y_score)

        fpr, tpr, _ = roc_curve(y_true.ravel(), y_score.ravel())
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.title(f"ROC - {split_name} | T={temperature}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(
            RESULTS_DIR, f"{split_name}_T{temperature}_roc.png"))
        plt.close()

# ------------------------------------------------------
# SAVE FINAL CSV
# ------------------------------------------------------
summary_df = pd.DataFrame(summary_rows)
csv_path = os.path.join(RESULTS_DIR, "summary_metrics.csv")
summary_df.round(3).to_csv(csv_path, index=False)

print(f"\nAll results saved to {csv_path}")
