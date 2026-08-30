"""
Trains a category-only baseline (fault_class predicted from MVTec category
alone) and compares it against: 
(1) the standalone image-only model, 
(2) the fusion model's image branch, isolated by corrupting its vibration input (the
same technique used in modality_shuffle_test.py) to observe how the branch
behaves as actually trained within 'both', 
(3) a per-category breakdown of whether 'both' repeats the category-only baseline's 
specific mistakes.
"""
import json

import pandas as pd
import torch
from torch.utils.data import DataLoader

from defect_detection.data.dataset import MultimodalDefectDataset
from defect_detection.evaluation.category_baseline import (
    compare_predictions_by_category,
    predict_category_only_baseline,
    print_category_comparison,
    train_category_only_baseline,
)
from defect_detection.evaluation.modality_shuffle import collect_predictions_with_corruption
from defect_detection.mlflow_utils import load_model_and_stats
from defect_detection.utils import find_project_root, load_yaml_config

if __name__ == "__main__":
    project_root = find_project_root()
    data_config = load_yaml_config("config/data_config.yaml")
    manifest_dir = project_root / data_config["paths"]["manifest_dir"]
    reports_dir = project_root / data_config["paths"]["reports_dir"]
    class_names = [ft["name"] for ft in data_config["cwru"]["fault_types"]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df = pd.read_csv(manifest_dir / "train.csv")
    test_df = pd.read_csv(manifest_dir / "test.csv")

    # Macro-F1 / per-class comparison across the four sources 
    category_baseline = train_category_only_baseline(train_df, test_df, class_names)

    with open(reports_dir / "evaluation" / "image" / "metrics.json") as f:
        image_only = json.load(f)
    with open(reports_dir / "modality_shuffle" / "results_zero.json") as f:
        shuffle_zero = json.load(f)
    with open(reports_dir / "modality_shuffle" / "results_shuffle.json") as f:
        shuffle_perm = json.load(f)

    print("Fault-type F1 comparison:")
    print_category_comparison(
        category_baseline,
        image_only["fault_metrics"],
        shuffle_zero["vibration_corrupted"]["fault_metrics"],
        shuffle_perm["vibration_corrupted"]["fault_metrics"],
    )

    # Per-category breakdown: test whether `both` repeats the category-only baseline's mistakes
    model, vib_mean, vib_std = load_model_and_stats("both", device=device)
    test_dataset = MultimodalDefectDataset(
        test_df, window_size=data_config["window_size"], fs=data_config["cwru"]["sampling_rate_hz"],
        training=False, vib_mean=vib_mean, vib_std=vib_std,
    )
    loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    both_predictions = collect_predictions_with_corruption(
        model, loader, device, "vibration", method="shuffle", seed=42,
    )
    _, category_pred = predict_category_only_baseline(train_df, test_df, class_names)

    breakdown = compare_predictions_by_category(
        test_df, both_predictions["fault_class_true"], both_predictions["fault_class_pred"], category_pred,
    )
    print("\nPer-category breakdown (both vs. category-only):")
    print(breakdown.to_string())