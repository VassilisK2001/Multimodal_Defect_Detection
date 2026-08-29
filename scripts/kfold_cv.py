import argparse
import logging

import pandas as pd
import torch

from defect_detection.evaluation.cross_validation import aggregate_cv_results, run_kfold_cv
from defect_detection.evaluation.reporting import save_cv_results
from defect_detection.utils import find_project_root, load_yaml_config

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                         force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=["both", "image", "vibration", "all"], default="all")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    modalities = ("both", "image", "vibration") if args.modality == "all" else (args.modality,)

    project_root = find_project_root()
    data_config = load_yaml_config("config/data_config.yaml")

    manifest_df = pd.read_csv(project_root / data_config["paths"]["manifest_dir"] / "manifest.csv")
    class_names = [ft["name"] for ft in data_config["cwru"]["fault_types"]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = project_root / data_config["paths"]["reports_dir"] / "cv_evaluation" / f"seed{args.seed}"

    for modality in modalities:
        logger.info("Running %d-fold CV for '%s' (seed=%d, %d training runs)...",
                    args.k, modality, args.seed, args.k)

        fold_results = run_kfold_cv(
            modality=modality, manifest_df=manifest_df,
            window_size=data_config["window_size"], fs=data_config["cwru"]["sampling_rate_hz"],
            class_names=class_names, k=args.k, seed=args.seed, device=device,
        )
        aggregated = aggregate_cv_results(fold_results)
        save_cv_results(modality, fold_results, aggregated, output_dir)

        defect_f1 = aggregated["defect_metrics"]["defect"]["f1"]
        fault_macro_f1 = aggregated["fault_metrics"]["macro_f1"]
        logger.info("  defect F1: %.3f \u00b1 %.3f", defect_f1["mean"], defect_f1["std"])
        logger.info("  fault macro-F1: %.3f \u00b1 %.3f", fault_macro_f1["mean"], fault_macro_f1["std"])

    logger.info("Results saved to %s", output_dir)