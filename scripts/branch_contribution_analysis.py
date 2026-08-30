"""
Computes exact 2-player Shapley branch-contribution (image vs. vibration)
analysis for the fusion model, both heads, on the test set. Each attribution
is averaged over k real background samples rather than a single reference
point, avoiding single-baseline sensitivity.
"""

import logging
from typing import cast

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from defect_detection.data.dataset import MultimodalDefectDataset
from defect_detection.data.features import extract_raw_vib_features_from_df
from defect_detection.data.normalization import apply_vibration_normalization
from defect_detection.data.image_io import load_images_for_df
from defect_detection.evaluation.predictions import collect_test_predictions
from defect_detection.interpretability.branch_contribution import (
    check_shapley_additivity_sample,
    compute_branch_contributions,
    prepare_background_samples,
)
from defect_detection.interpretability.shap_explain import (
    select_head1_background_rows,
    select_head2_background_rows,
)
from defect_detection.interpretability.visualization import (
    plot_branch_contribution_scatter,
    plot_branch_contribution_violin,
)
from defect_detection.mlflow_utils import load_model_and_stats
from defect_detection.utils import find_project_root, load_yaml_config


logger = logging.getLogger(__name__)

N_TEST = 200
K_BACKGROUND = 30


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                         force=True)

    project_root = find_project_root()
    data_config = load_yaml_config("config/data_config.yaml")
    train_df = pd.read_csv(project_root / data_config["paths"]["manifest_dir"] / "train.csv")
    test_df = pd.read_csv(project_root / data_config["paths"]["manifest_dir"] / "test.csv")
    class_names = [ft["name"] for ft in data_config["cwru"]["fault_types"]]
    window_size, fs = data_config["window_size"], data_config["cwru"]["sampling_rate_hz"]
    output_dir = project_root / data_config["paths"]["reports_dir"] / "branch_contribution"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading fusion model...")
    model, vib_mean, vib_std = load_model_and_stats("both", device=torch.device("cpu"))
    model.eval()

    logger.info("Preparing background samples...")
    head1_bg_rows = select_head1_background_rows(train_df, n_samples=50)
    head2_bg_rows = select_head2_background_rows(train_df, n_samples=50)
    head1_backgrounds = prepare_background_samples(
        head1_bg_rows, project_root, window_size, fs, vib_mean, vib_std, k=K_BACKGROUND)
    head2_backgrounds = prepare_background_samples(
        head2_bg_rows, project_root, window_size, fs, vib_mean, vib_std, k=K_BACKGROUND)

    logger.info("Preparing test data...")
    test_subset = test_df.sample(n=min(N_TEST, len(test_df)), random_state=42)
    test_images = load_images_for_df(test_subset, project_root)
    test_raw = extract_raw_vib_features_from_df(test_subset, window_size, fs)
    test_norm = torch.tensor(apply_vibration_normalization(test_raw, vib_mean, vib_std), dtype=torch.float32)

    logger.info("Checking Shapley additivity (defect gate)...")
    additivity_result = check_shapley_additivity_sample(
        model, "defect", test_images[:3], test_norm[:3], head1_backgrounds)
    logger.info("Additivity: max_residual=%.6f, within_tolerance=%s",
                additivity_result["max_residual"], additivity_result["within_tolerance"])

    logger.info("Computing branch contributions: defect gate...")
    phi_image_defect, phi_vib_defect, se_info_defect = compute_branch_contributions(
        model, "defect", test_images, test_norm, head1_backgrounds)
    logger.info(
        "Defect gate SE (K=%d): median se_image=%.4f, median se_vib=%.4f (vs. median |phi_vib|=%.4f)",
        se_info_defect["k"], np.median(se_info_defect["se_image"]), np.median(se_info_defect["se_vib"]),
        np.median(np.abs(phi_vib_defect)),
    )

    test_dataset = MultimodalDefectDataset(
        test_subset, window_size=window_size, fs=fs, training=False, vib_mean=vib_mean, vib_std=vib_std)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    predictions = collect_test_predictions(model, test_loader, device=torch.device("cpu"))

    labels = np.full(len(test_subset), "correct_normal", dtype=object)
    y_true, y_pred = predictions["is_defect_true"], predictions["is_defect_pred"]
    labels[(y_true == 1) & (y_pred == 1)] = "correct_defective"
    labels[(y_true == 1) & (y_pred == 0)] = "false_negative"
    labels[(y_true == 0) & (y_pred == 1)] = "false_positive"

    fig_defect = plot_branch_contribution_scatter(
        phi_image_defect, phi_vib_defect, labels, title="Branch Contribution: Defect Gate")
    fig_defect.savefig(output_dir / "branch_contribution_defect_gate.png", dpi=150, bbox_inches="tight")

    pct_vib_dominant_defect = float(np.mean(np.abs(phi_vib_defect) > np.abs(phi_image_defect)) * 100)
    logger.info("Defect gate: %.1f%% of samples have |vib| > |image| contribution.", pct_vib_dominant_defect)

    logger.info("Computing branch contributions: fault type...")
    defective_subset = cast(pd.DataFrame, test_subset[test_subset.is_defect == 1])
    defective_images = load_images_for_df(defective_subset, project_root)
    defective_raw = extract_raw_vib_features_from_df(defective_subset, window_size, fs)
    defective_norm = torch.tensor(
        apply_vibration_normalization(defective_raw, vib_mean, vib_std), dtype=torch.float32)

    phi_image_per_class = {}
    phi_vib_per_class = {}
    for class_idx, class_name in enumerate(class_names):
        phi_image_c, phi_vib_c, se_info_c = compute_branch_contributions(
            model, "fault", defective_images, defective_norm, head2_backgrounds, target_class=class_idx)
        phi_image_per_class[class_name] = phi_image_c
        phi_vib_per_class[class_name] = phi_vib_c
        pct_vib_dominant = float(np.mean(np.abs(phi_vib_c) > np.abs(phi_image_c)) * 100)
        logger.info("%s: %.1f%% of samples have |vib| > |image| contribution (SE: image=%.4f, vib=%.4f).",
                    class_name, pct_vib_dominant, np.median(se_info_c["se_image"]), np.median(se_info_c["se_vib"]))

    fig_fault = plot_branch_contribution_violin(
        phi_image_per_class, phi_vib_per_class, class_names, title="Branch Contribution: Fault Type")
    fig_fault.savefig(output_dir / "branch_contribution_fault_type.png", dpi=150, bbox_inches="tight")

    logger.info("Branch contribution analysis complete. Figures saved to %s", output_dir)