
import pandas as pd
import torch

from defect_detection.evaluation.reporting import save_evaluation_results
from defect_detection.evaluation.run_evaluation import evaluate_model
from defect_detection.utils import find_project_root, load_yaml_config

if __name__ == "__main__":
    project_root = find_project_root()
    data_config = load_yaml_config("config/data_config.yaml")

    test_df = pd.read_csv(project_root / data_config["paths"]["manifest_dir"] / "test.csv")
    class_names = [ft["name"] for ft in data_config["cwru"]["fault_types"]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = project_root / data_config["paths"]["reports_dir"] / "evaluation"

    summary_rows = []

    for modality in ("both", "image", "vibration"):
        print(f"Evaluating '{modality}'...")

        result = evaluate_model(
            modality, test_df,
            window_size=data_config["window_size"],
            fs=data_config["cwru"]["sampling_rate_hz"],
            class_names=class_names,
            device=device,
        )

        save_evaluation_results(modality, result, output_dir)

        summary_rows.append({
            "modality": modality,
            "defect_precision": result["defect_metrics"]["defect"]["precision"],
            "defect_recall": result["defect_metrics"]["defect"]["recall"],
            "defect_f1": result["defect_metrics"]["defect"]["f1"],
            "fault_macro_f1": result["fault_metrics"]["macro_f1"],
        })

    summary_df = pd.DataFrame(summary_rows)
    print("\n=== Summary (test set) ===")
    print(summary_df.to_string(index=False))

    summary_df.to_csv(output_dir / "summary.csv", index=False)
    print(f"\nFull results saved to {output_dir}")