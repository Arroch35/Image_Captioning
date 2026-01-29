import numpy as np
from scipy.stats import ttest_rel

# ------------------------------------------------------
# LOAD FOLD-WISE TOP-1 ACCURACIES
# ------------------------------------------------------
base_dir = "../data/results/part1"   

best_model = "swin_tiny_bert-base-uncased"
other_models = [
    "resnet50_bert-base-uncased",
    "resnet50_gpt2",
    "swin_tiny_gpt2",
]

best_scores = np.load(f"{base_dir}/{best_model}/custom_clip_top1_folds.npy")

print(f"\nBest model: {best_model}")
print(f"Fold-wise Top-1: {best_scores}\n")

# ------------------------------------------------------
# PAIRED T-TESTS
# ------------------------------------------------------
alpha = 0.05

for model in other_models:
    scores = np.load(f"{base_dir}/{model}/custom_clip_top1_folds.npy")

    t_stat, p_value = ttest_rel(best_scores, scores)

    print(f"Comparison: {best_model} vs {model}")
    print(f"  t-statistic = {t_stat:.4f}")
    print(f"  p-value     = {p_value:.6f}")

    if p_value < alpha:
        print(" Statistically significant difference (p < 0.05)\n")
    else:
        print(" No statistically significant difference (p = 0.05)\n")