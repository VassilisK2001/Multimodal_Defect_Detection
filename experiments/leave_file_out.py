
import pandas as pd
import torch
from torch.utils.data import DataLoader

from defect_detection.data.dataset import MultimodalDefectDataset
from defect_detection.data.splitting import build_leave_file_out_split, select_files_to_hold_out
from defect_detection.evaluation.metrics import compute_defect_gate_metrics, compute_fault_type_metrics
from defect_detection.evaluation.predictions import collect_test_predictions
from defect_detection.training.train import train_from_dataframes
from defect_detection.utils import find_project_root, load_yaml_config


def print_defect_metrics(defect_metrics: dict) -> None:
    print("\nDefect gate (held-out file):")
    print(f"{'':10}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}")
    for label in ("normal", "defect"):
        m = defect_metrics[label]
        print(f"{label:10}{m['precision']:>10.3f}{m['recall']:>10.3f}{m['f1']:>10.3f}{m['support']:>10}")


def print_fault_type_metrics(fault_metrics: dict) -> None:
    print("\nFault-type (held-out file, defective samples only):")
    print(f"{'':14}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}")
    for fault_class, m in fault_metrics["per_class"].items():
        print(f"{fault_class:14}{m['precision']:>10.3f}{m['recall']:>10.3f}{m['f1']:>10.3f}{m['support']:>10}")
    print(f"{'macro_f1':14}{fault_metrics['macro_f1']:>10.3f}")


if __name__ == "__main__":
    project_root = find_project_root()
    data_config = load_yaml_config("config/data_config.yaml")
    manifest_df = pd.read_csv(project_root / data_config["paths"]["manifest_dir"] / "manifest.csv")
    class_names = [ft["name"] for ft in data_config["cwru"]["fault_types"]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    held_out_files = select_files_to_hold_out(manifest_df, seed=42)
    print(f"Held-out files: {held_out_files}")

    train_df, val_df, held_out_test_df = build_leave_file_out_split(manifest_df, held_out_files, seed=42)

    model, vib_mean, vib_std = train_from_dataframes(
        train_df, val_df, modality="vibration", seed=42,
        run_name_suffix="_leave_file_out", register_model=False,
    )

    test_dataset = MultimodalDefectDataset(
        held_out_test_df, window_size=data_config["window_size"],
        fs=data_config["cwru"]["sampling_rate_hz"], training=False,
        vib_mean=vib_mean, vib_std=vib_std,
    )
    predictions = collect_test_predictions(model, DataLoader(test_dataset, batch_size=32), device=device)

    defect_metrics = compute_defect_gate_metrics(predictions["is_defect_true"], predictions["is_defect_pred"])
    fault_metrics = compute_fault_type_metrics(
        predictions["fault_class_true"], predictions["fault_class_pred"], class_names,
    )

    print_defect_metrics(defect_metrics)
    print_fault_type_metrics(fault_metrics)