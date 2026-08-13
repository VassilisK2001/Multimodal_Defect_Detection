
import argparse

import pandas as pd
import torch

from defect_detection.evaluation.modality_shuffle import run_modality_shuffle_test
from defect_detection.evaluation.reporting import save_modality_shuffle_results
from defect_detection.mlflow_utils import load_model_and_stats
from defect_detection.utils import find_project_root, load_yaml_config


def print_comparison(results: dict) -> None:
    print(f"\n{'':20}{'defect F1':>12}{'defect recall':>16}{'fault macro-F1':>18}")
    for label, result in results.items():
        defect = result["defect_metrics"]["defect"]
        fault_macro_f1 = result["fault_metrics"]["macro_f1"]
        print(f"{label:20}{defect['f1']:>12.3f}{defect['recall']:>16.3f}{fault_macro_f1:>18.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["shuffle", "zero"], default="shuffle")
    parser.add_argument("--corrupt-modality", choices=["image", "vibration", "both"], default="both")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    corrupt_modalities = ("image", "vibration") if args.corrupt_modality == "both" else (args.corrupt_modality,)

    project_root = find_project_root()
    data_config = load_yaml_config("config/data_config.yaml")

    test_df = pd.read_csv(project_root / data_config["paths"]["manifest_dir"] / "test.csv")
    class_names = [ft["name"] for ft in data_config["cwru"]["fault_types"]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, vib_mean, vib_std = load_model_and_stats("both", device=device)

    results = run_modality_shuffle_test(
        model, test_df, window_size=data_config["window_size"],
        fs=data_config["cwru"]["sampling_rate_hz"], class_names=class_names,
        vib_mean=vib_mean, vib_std=vib_std, device=device, method=args.method,
        corrupt_modalities=corrupt_modalities, seed=args.seed,
    )

    output_dir = project_root / data_config["paths"]["reports_dir"] / "modality_shuffle"
    save_modality_shuffle_results(results, output_dir, method=args.method)
    print(f"Full results (all metrics, all classes) saved to {output_dir}")

    print_comparison(results)