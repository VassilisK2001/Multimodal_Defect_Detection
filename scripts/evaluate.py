"""
Evaluates three separately trained models, one receiving both image and
vibration inputs ('both'), one receiving only images ('image'), and one
receiving only vibration signal data ('vibration') on the test set, 
saving per-model results and a summary CSV.
"""

import logging

import pandas as pd
import torch

from defect_detection.evaluation.reporting import save_evaluation_results
from defect_detection.evaluation.run_evaluation import evaluate_model
from defect_detection.utils import find_project_root, load_yaml_config

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                         force=True)

    project_root = find_project_root()
    data_config = load_yaml_config("config/data_config.yaml")

    test_df = pd.read_csv(project_root / data_config["paths"]["manifest_dir"] / "test.csv")
    class_names = [ft["name"] for ft in data_config["cwru"]["fault_types"]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = project_root / data_config["paths"]["reports_dir"] / "evaluation"

    summary_rows = []

    for modality in ("both", "image", "vibration"):
        logger.info("Evaluating '%s'...", modality)

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
    logger.info("=== Summary (test set) ===\n%s", summary_df.to_string(index=False))

    summary_df.to_csv(output_dir / "summary.csv", index=False)
    logger.info("Full results saved to %s", output_dir)