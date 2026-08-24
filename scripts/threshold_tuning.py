import json
import logging

import pandas as pd
import torch
from torch.utils.data import DataLoader

from defect_detection.data.dataset import MultimodalDefectDataset
from defect_detection.evaluation.metrics import compute_defect_gate_metrics, compute_fault_type_metrics
from defect_detection.evaluation.predictions import collect_test_predictions
from defect_detection.evaluation.visualization import (
    plot_bootstrap_distribution,
    plot_defect_gate_confusion_matrix,
    plot_fault_type_confusion_matrix,
)
from defect_detection.inference.thresholds import bootstrap_threshold_distribution
from defect_detection.mlflow_utils import load_model_and_stats
from defect_detection.utils import find_project_root, load_yaml_config


logger = logging.getLogger(__name__)

TARGET_RECALL = 0.95
N_BOOTSTRAP = 1000


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                         force=True)

    project_root = find_project_root()
    data_config = load_yaml_config("config/data_config.yaml")
    val_df = pd.read_csv(project_root / data_config["paths"]["manifest_dir"] / "val.csv")
    test_df = pd.read_csv(project_root / data_config["paths"]["manifest_dir"] / "test.csv")
    class_names = [ft["name"] for ft in data_config["cwru"]["fault_types"]]
    window_size, fs = data_config["window_size"], data_config["cwru"]["sampling_rate_hz"]
    output_dir = project_root / data_config["paths"]["reports_dir"] / "final_report"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading final deployed model...")
    model, vib_mean, vib_std = load_model_and_stats("both", device=device)

    logger.info("Collecting validation-set predictions...")
    val_dataset = MultimodalDefectDataset(
        val_df, window_size=window_size, fs=fs, training=False, vib_mean=vib_mean, vib_std=vib_std)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    val_predictions = collect_test_predictions(model, val_loader, device=device)

    logger.info("Running stratified bootstrap threshold estimation (n_bootstrap=%d, target_recall=%.2f)...",
                N_BOOTSTRAP, TARGET_RECALL)
    bootstrap_result = bootstrap_threshold_distribution(
        val_predictions["is_defect_true"], val_predictions["defect_proba"],
        target_recall=TARGET_RECALL, n_bootstrap=N_BOOTSTRAP,
    )
    logger.info(
        "Bootstrap threshold: mean=%.4f (std=%.4f) | recall: mean=%.4f (std=%.4f) | "
        "precision: mean=%.4f (std=%.4f) | target achieved in %d/%d resamples",
        bootstrap_result["mean_threshold"], bootstrap_result["std_threshold"],
        bootstrap_result["mean_recall"], bootstrap_result["std_recall"],
        bootstrap_result["mean_precision"], bootstrap_result["std_precision"],
        bootstrap_result["n_target_achieved"], bootstrap_result["n_bootstrap"],
    )

    final_threshold = bootstrap_result["mean_threshold"]
    logger.info("Final threshold selected (bootstrap mean): %.4f", final_threshold)

    logger.info("Plotting bootstrap distributions...")
    fig_threshold_dist = plot_bootstrap_distribution(
        bootstrap_result["thresholds"], bootstrap_result["mean_threshold"], bootstrap_result["std_threshold"],
        xlabel="Threshold", title="Bootstrap Distribution: Selected Threshold",
    )
    fig_threshold_dist.savefig(output_dir / "bootstrap_threshold_distribution.png", dpi=150, bbox_inches="tight")

    fig_recall_dist = plot_bootstrap_distribution(
        bootstrap_result["recalls"], bootstrap_result["mean_recall"], bootstrap_result["std_recall"],
        xlabel="Recall", title="Bootstrap Distribution: Achieved Recall",
    )
    fig_recall_dist.savefig(output_dir / "bootstrap_recall_distribution.png", dpi=150, bbox_inches="tight")

    fig_precision_dist = plot_bootstrap_distribution(
        bootstrap_result["precisions"], bootstrap_result["mean_precision"], bootstrap_result["std_precision"],
        xlabel="Precision", title="Bootstrap Distribution: Achieved Precision",
    )
    fig_precision_dist.savefig(output_dir / "bootstrap_precision_distribution.png", dpi=150, bbox_inches="tight")

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
            "bootstrap": {
                "mean_threshold": bootstrap_result["mean_threshold"],
                "std_threshold": bootstrap_result["std_threshold"],
                "mean_recall": bootstrap_result["mean_recall"],
                "std_recall": bootstrap_result["std_recall"],
                "mean_precision": bootstrap_result["mean_precision"],
                "std_precision": bootstrap_result["std_precision"],
                "n_target_achieved": bootstrap_result["n_target_achieved"],
                "n_bootstrap": bootstrap_result["n_bootstrap"],
            },
            "defect_metrics": defect_metrics,
            "fault_metrics": fault_metrics,
        }, f, indent=2, default=str)

    logger.info("Final report complete. Results saved to %s", output_dir)