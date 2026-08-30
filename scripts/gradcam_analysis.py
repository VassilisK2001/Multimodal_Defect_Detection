"""
Grad-CAM analysis of the fusion model's ('both') image branch
(convolutional encoder), on the test set:
- Defect gate: correct-defective / false-negative / false-positive
- Fault type: correct / misclassified, per class, explaining the predicted
  class
"""

import pandas as pd
import torch
from torch.utils.data import DataLoader

from defect_detection.data.dataset import MultimodalDefectDataset
from defect_detection.evaluation.predictions import collect_test_predictions
from defect_detection.interpretability.example_selection import (
    find_defect_gate_examples,
    find_fault_type_examples,
)
from defect_detection.interpretability.visualization import (
    plot_defect_gate_gradcam_grid,
    plot_fault_type_gradcam_grid,
)
from defect_detection.mlflow_utils import load_model_and_stats
from defect_detection.utils import find_project_root, load_yaml_config


if __name__ == "__main__":
    project_root = find_project_root()
    data_config = load_yaml_config("config/data_config.yaml")
    test_df = pd.read_csv(project_root / data_config["paths"]["manifest_dir"] / "test.csv")
    class_names = [ft["name"] for ft in data_config["cwru"]["fault_types"]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = project_root / data_config["paths"]["reports_dir"] / "gradcam"
    output_dir.mkdir(parents=True, exist_ok=True)

    model, vib_mean, vib_std = load_model_and_stats("both", device=device)
    dataset = MultimodalDefectDataset(
        test_df, window_size=data_config["window_size"], fs=data_config["cwru"]["sampling_rate_hz"],
        training=False, vib_mean=vib_mean, vib_std=vib_std,
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    predictions = collect_test_predictions(model, loader, device=device)

    defect_examples = find_defect_gate_examples(predictions, n=3)
    fig_defect = plot_defect_gate_gradcam_grid(model, dataset, defect_examples, n=3)
    fig_defect.savefig(output_dir / "defect_gate_gradcam.png", dpi=150, bbox_inches="tight")

    fault_examples = find_fault_type_examples(predictions, class_names, n=3)
    fig_fault = plot_fault_type_gradcam_grid(model, dataset, fault_examples, class_names, n=3)
    fig_fault.savefig(output_dir / "fault_type_gradcam.png", dpi=150, bbox_inches="tight")

    print(f"Figures saved to {output_dir}")