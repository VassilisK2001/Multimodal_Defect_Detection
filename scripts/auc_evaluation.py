import logging

import pandas as pd
import torch

from defect_detection.evaluation.run_auc_evaluation import (
    compute_global_oof_metrics,
    log_fold_level_summary,
    log_global_oof_metrics,
    run_oof_cross_validation,
)
from defect_detection.evaluation.reporting import save_oof_results
from defect_detection.evaluation.visualization import (
    plot_defect_gate_global_curves,
    plot_fault_type_global_pr_curves,
)
from defect_detection.utils import find_project_root, load_yaml_config


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,)
    logger = logging.getLogger(__name__)

    project_root = find_project_root()
    data_config = load_yaml_config("config/data_config.yaml")
    manifest_df = pd.read_csv(project_root / data_config["paths"]["manifest_dir"] / "manifest.csv")
    class_names = [ft["name"] for ft in data_config["cwru"]["fault_types"]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = project_root / data_config["paths"]["reports_dir"] / "oof_evaluation"

    logger.info("Step 1-2/6: initializing OOF arrays and running k-fold cross-validation...")
    oof_results = run_oof_cross_validation(
        manifest_df, class_names,
        window_size=data_config["window_size"], fs=data_config["cwru"]["sampling_rate_hz"],
        k=3, seed=42, device=device,
    )
    logger.info("Step 1-2/6 complete.")

    logger.info("Step 3/6: fold-level summary")
    log_fold_level_summary(oof_results, class_names)
    logger.info("Step 3/6 complete.")

    logger.info("Step 4/6: computing global OOF ROC/PR AUC...")
    global_metrics = compute_global_oof_metrics(oof_results, class_names)
    log_global_oof_metrics(global_metrics, class_names)
    logger.info("Step 4/6 complete.")

    logger.info("Step 5/6: plotting defect gate ROC/PR curves ...")
    fig_a = plot_defect_gate_global_curves(oof_results, global_metrics)
    logger.info("Step 5/6 complete.")

    logger.info("Step 6/6: plotting PR curves for fault-type classifier ... ")
    fig_b = plot_fault_type_global_pr_curves(
        oof_results, global_metrics, class_names,
        subplot_order=["inner_race", "outer_race", "ball"],
    )
    logger.info("Step 6/6 complete.")

    save_oof_results(
        oof_results, global_metrics,
        figures={"defect_gate_curves": fig_a, "fault_type_curves": fig_b},
        output_dir=output_dir,
    )
    logger.info("All steps complete. Results and figures saved to %s", output_dir)