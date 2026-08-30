""" 
Full test-set evaluation for a single registered model
"""



import torch
from torch.utils.data import DataLoader

from defect_detection.data.dataset import MultimodalDefectDataset
from defect_detection.evaluation.metrics import compute_defect_gate_metrics, compute_fault_type_metrics
from defect_detection.evaluation.predictions import collect_test_predictions
from defect_detection.evaluation.visualization import (
    plot_defect_gate_confusion_matrix,
    plot_fault_type_confusion_matrix,
)
from defect_detection.mlflow_utils import load_model_and_stats


def evaluate_model(modality: str, test_df, window_size: int, fs: int,
                    class_names: list[str], device: torch.device = torch.device("cpu"),
                    version: str = "latest") -> dict:
    """Full test-set evaluation for one registered model.

    Args:
        modality: "both", "image", or "vibration".
        test_df: Test split manifest.
        window_size: Vibration window size in samples.
        fs: Vibration sampling rate in Hz.
        class_names: Fault class names, in index order.
        device: Device to run inference on.
        version: Registered model version, or "latest".

    Returns:
        Dict with 'defect_metrics', 'fault_metrics', 'figures'.
    """
    model, vib_mean, vib_std = load_model_and_stats(modality, version=version, device=device)

    test_dataset = MultimodalDefectDataset(
        test_df, window_size=window_size, fs=fs, training=False,
        vib_mean=vib_mean, vib_std=vib_std,
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    predictions = collect_test_predictions(model, test_loader, device=device)

    defect_metrics = compute_defect_gate_metrics(
        predictions["is_defect_true"], predictions["is_defect_pred"],
    )
    fault_metrics = compute_fault_type_metrics(
        predictions["fault_class_true"], predictions["fault_class_pred"], class_names,
    )

    figures = {
        "defect_confusion_matrix": plot_defect_gate_confusion_matrix(
            predictions["is_defect_true"], predictions["is_defect_pred"],
        ),
        "fault_type_confusion_matrix": plot_fault_type_confusion_matrix(
            predictions["fault_class_true"], predictions["fault_class_pred"], class_names,
        ),
    }

    return {"defect_metrics": defect_metrics, "fault_metrics": fault_metrics, "figures": figures}