import json
import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from defect_detection.data.dataset import MultimodalDefectDataset
from defect_detection.evaluation.metrics import compute_defect_gate_metrics, compute_fault_type_metrics
from defect_detection.evaluation.predictions import collect_test_predictions
from defect_detection.evaluation.run_auc_evaluation import run_oof_cross_validation
from defect_detection.evaluation.visualization import (
    plot_defect_gate_confusion_matrix,
    plot_fault_type_confusion_matrix,
    plot_pr_curve_with_threshold,
)
from defect_detection.inference.thresholds import (
    check_threshold_transfers_to_final_model,
    select_recall_constrained_threshold,
)
from defect_detection.mlflow_utils import load_model_and_stats
from defect_detection.utils import find_project_root, load_yaml_config


logger = logging.getLogger(__name__)

TARGET_RECALL = 0.95


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                         force=True)

    project_root = find_project_root()
    data_config = load_yaml_config("config/data_config.yaml")
    manifest_df = pd.read_csv(project_root / data_config["paths"]["manifest_dir"] / "manifest.csv")
    val_df = pd.read_csv(project_root / data_config["paths"]["manifest_dir"] / "val.csv")
    test_df = pd.read_csv(project_root / data_config["paths"]["manifest_dir"] / "test.csv")
    class_names = [ft["name"] for ft in data_config["cwru"]["fault_types"]]
    window_size, fs = data_config["window_size"], data_config["cwru"]["sampling_rate_hz"]
    output_dir = project_root / data_config["paths"]["reports_dir"] / "final_report"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    logger.info("Running OOF cross-validation for 'both' (defect-gate threshold estimation)...")
    oof_results = run_oof_cross_validation(
        manifest_df, class_names, window_size, fs, k=5, seed=42, device=device, modalities=("both",),
    )
    oof_defect_true = oof_results["is_defect_true"]
    oof_defect_proba = oof_results["models"]["both"]["oof_defect_proba"]
    np.savez(output_dir / "oof_defect_predictions.npz", y_true=oof_defect_true, y_proba=oof_defect_proba)


    logger.info("Selecting recall-constrained threshold (target_recall=%.2f)...", TARGET_RECALL)
    threshold_result = select_recall_constrained_threshold(oof_defect_true, oof_defect_proba, TARGET_RECALL)
    logger.info("OOF-derived threshold: %.4f (precision=%.4f, recall=%.4f, target_recall_achieved=%s)",
                threshold_result["threshold"], threshold_result["precision"],
                threshold_result["recall"], threshold_result["target_recall_achieved"])

    fig_pr = plot_pr_curve_with_threshold(
        oof_defect_true, oof_defect_proba, threshold_result["threshold"],
        threshold_result["precision"], threshold_result["recall"],
        title="Defect Gate: OOF PR Curve with Selected Threshold",
    )
    fig_pr.savefig(output_dir / "pr_curve_with_threshold.png", dpi=150, bbox_inches="tight")


    logger.info("Loading final deployed model and checking threshold transfer on val.csv...")
    model, vib_mean, vib_std = load_model_and_stats("both", device=device)
    val_dataset = MultimodalDefectDataset(
        val_df, window_size=window_size, fs=fs, training=False, vib_mean=vib_mean, vib_std=vib_std)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    val_predictions = collect_test_predictions(model, val_loader, device=device)

    transfer_check = check_threshold_transfers_to_final_model(
        threshold_result["threshold"], val_predictions["is_defect_true"], val_predictions["defect_proba"],
        TARGET_RECALL,
    )
    logger.info("Validation-set transfer check: recall=%.4f, precision=%.4f, diverges=%s",
                transfer_check["recall"], transfer_check["precision"], transfer_check["diverges"])

    
    final_threshold = threshold_result["threshold"]
    if transfer_check["diverges"]:
        logger.warning(
            "OOF-derived threshold diverges on the final model's validation predictions "
            "(recall=%.4f vs. target=%.2f) — falling back to a validation-set-only threshold.",
            transfer_check["recall"], TARGET_RECALL,
        )
        fallback_result = select_recall_constrained_threshold(
            val_predictions["is_defect_true"], val_predictions["defect_proba"], TARGET_RECALL,
        )
        final_threshold = fallback_result["threshold"]
        logger.info("Fallback validation-set threshold: %.4f", final_threshold)
    logger.info("Final threshold selected: %.4f", final_threshold)


    logger.info("Generating final report on test.csv...")
    test_dataset = MultimodalDefectDataset(
        test_df, window_size=window_size, fs=fs, training=False, vib_mean=vib_mean, vib_std=vib_std)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    test_predictions = collect_test_predictions(
        model, test_loader, device=device, defect_threshold=final_threshold,
    )

    defect_metrics = compute_defect_gate_metrics(
        test_predictions["is_defect_true"], test_predictions["is_defect_pred"])
    fault_metrics = compute_fault_type_metrics(
        test_predictions["fault_class_true"], test_predictions["fault_class_pred"], class_names)

    logger.info("Final defect-gate metrics (test.csv, threshold=%.4f): %s", final_threshold, defect_metrics)
    logger.info("Final fault-type macro F1 (test.csv): %.4f", fault_metrics["macro_f1"])

    fig_defect_cm = plot_defect_gate_confusion_matrix(
        test_predictions["is_defect_true"], test_predictions["is_defect_pred"])
    fig_defect_cm.savefig(output_dir / "confusion_matrix_defect_gate.png", dpi=150, bbox_inches="tight")

    fig_fault_cm = plot_fault_type_confusion_matrix(
        test_predictions["fault_class_true"], test_predictions["fault_class_pred"], class_names)
    fig_fault_cm.savefig(output_dir / "confusion_matrix_fault_type.png", dpi=150, bbox_inches="tight")


    with open(output_dir / "final_report.json", "w", encoding="utf-8") as f:
        json.dump({
            "final_threshold": final_threshold,
            "oof_threshold_selection": threshold_result,
            "validation_transfer_check": transfer_check,
            "defect_metrics": defect_metrics,
            "fault_metrics": fault_metrics,
        }, f, indent=2, default=str)

    logger.info("Final report complete. Results saved to %s", output_dir)