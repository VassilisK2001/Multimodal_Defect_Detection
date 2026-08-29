import logging
from typing import cast

import pandas as pd
import torch

from defect_detection.data.image_io import load_images_for_df
from defect_detection.data.features import extract_raw_vib_features_from_df
from defect_detection.data.normalization import apply_vibration_normalization
from defect_detection.interpretability.example_selection import (
    find_vibration_fails_fusion_succeeds_examples,
)
from defect_detection.interpretability.shap_explain import (
    approximate_predictions_from_shap,
    check_additivity,
    check_additivity_per_instance,
    compute_shap_batched,
    compute_shap_per_instance,
    make_vib_predict_fn,
    select_head1_background_rows,
    select_head2_background_rows,
    select_top_features,
    summarize_background,
)
from defect_detection.interpretability.visualization import (
    plot_beeswarm_comparison,
    plot_dependence,
    plot_waterfall,
)
from defect_detection.mlflow_utils import load_model_and_stats
from defect_detection.utils import find_project_root, load_yaml_config


logger = logging.getLogger(__name__)

N_TEST = 200
N_BACKGROUND = 75


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                         force=True)

    project_root = find_project_root()
    data_config = load_yaml_config("config/data_config.yaml")
    train_df = pd.read_csv(project_root / data_config["paths"]["manifest_dir"] / "train.csv")
    test_df = pd.read_csv(project_root / data_config["paths"]["manifest_dir"] / "test.csv")
    class_names = [ft["name"] for ft in data_config["cwru"]["fault_types"]]
    window_size, fs = data_config["window_size"], data_config["cwru"]["sampling_rate_hz"]
    output_dir = project_root / data_config["paths"]["reports_dir"] / "shap_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading models...")
    vib_model, vib_mean_v, vib_std_v = load_model_and_stats("vibration", device=torch.device("cpu"))
    fusion_model, vib_mean_f, vib_std_f = load_model_and_stats("both", device=torch.device("cpu"))

    logger.info("Building backgrounds...")
    head1_bg_raw = extract_raw_vib_features_from_df(
        select_head1_background_rows(train_df, n_samples=N_BACKGROUND), window_size, fs)
    head2_bg_raw = extract_raw_vib_features_from_df(
        select_head2_background_rows(train_df, n_samples=N_BACKGROUND), window_size, fs)

    head1_bg_vib = summarize_background(apply_vibration_normalization(head1_bg_raw, vib_mean_v, vib_std_v))
    head1_bg_fusion = summarize_background(apply_vibration_normalization(head1_bg_raw, vib_mean_f, vib_std_f))
    head2_bg_vib = summarize_background(apply_vibration_normalization(head2_bg_raw, vib_mean_v, vib_std_v))
    head2_bg_fusion = summarize_background(apply_vibration_normalization(head2_bg_raw, vib_mean_f, vib_std_f))

    logger.info("Preparing test subsets...")
    test_subset = test_df.sample(n=min(N_TEST, len(test_df)), random_state=42)
    test_raw = extract_raw_vib_features_from_df(test_subset, window_size, fs)
    defective_subset = cast(pd.DataFrame, test_subset[test_subset.is_defect == 1])
    defective_raw = extract_raw_vib_features_from_df(defective_subset, window_size, fs)


    logger.info("Computing SHAP values: vibration-only, Head 1...")
    vib_defect_predict_fn = make_vib_predict_fn(vib_model, head="defect")
    test_norm_vib_h1 = apply_vibration_normalization(test_raw, vib_mean_v, vib_std_v)
    vib_shap_defect = compute_shap_batched(vib_defect_predict_fn, head1_bg_vib, test_norm_vib_h1, test_raw)

    logger.info("Computing SHAP values: vibration-only, Head 2...")
    vib_fault_predict_fn = make_vib_predict_fn(vib_model, head="fault")
    test_norm_vib_h2 = apply_vibration_normalization(defective_raw, vib_mean_v, vib_std_v)
    vib_shap_fault = compute_shap_batched(vib_fault_predict_fn, head2_bg_vib, test_norm_vib_h2, defective_raw)

    logger.info("Checking additivity (vibration-only)...")
    for name, shap_values, predict_fn, features in [
        ("defect", vib_shap_defect, vib_defect_predict_fn, test_norm_vib_h1),
        ("fault", vib_shap_fault, vib_fault_predict_fn, test_norm_vib_h2),
    ]:
        result = check_additivity(shap_values[:10], predict_fn, features[:10])
        logger.info("Additivity (vib_%s): max_residual=%.4f, within_tolerance=%s",
                    name, result["max_residual"], result["within_tolerance"])

    vib_is_defect_pred = (approximate_predictions_from_shap(vib_shap_defect) >= 0.5).astype(int)

    logger.info("Computing SHAP values: fusion, Head 1...")
    test_images = load_images_for_df(test_subset, project_root)
    test_norm_fusion_h1 = apply_vibration_normalization(test_raw, vib_mean_f, vib_std_f)
    fusion_shap_defect = compute_shap_per_instance(
        fusion_model, "defect", head1_bg_fusion, test_norm_fusion_h1, test_raw, test_images)

    logger.info("Computing SHAP values: fusion, Head 2...")
    defective_images = load_images_for_df(defective_subset, project_root)
    test_norm_fusion_h2 = apply_vibration_normalization(defective_raw, vib_mean_f, vib_std_f)
    fusion_shap_fault = compute_shap_per_instance(
        fusion_model, "fault", head2_bg_fusion, test_norm_fusion_h2, defective_raw, defective_images)

    logger.info("Checking additivity (fusion)...")
    for name, shap_values, images, features in [
        ("defect", fusion_shap_defect, test_images, test_norm_fusion_h1),
        ("fault", fusion_shap_fault, defective_images, test_norm_fusion_h2),
    ]:
        result = check_additivity_per_instance(fusion_model, name, shap_values, features, images)
        logger.info(
            "Additivity (fusion_%s): max_residual=%.4f, mean_residual=%.4f, %d/%d rows within tolerance",
            name, result["max_residual"], result["mean_residual"],
            result["n_within_tolerance"], result["n_total"],
        )

    fusion_is_defect_pred = (approximate_predictions_from_shap(fusion_shap_defect) >= 0.5).astype(int)


    logger.info("Plotting beeswarm comparisons...")
    fig_h1 = plot_beeswarm_comparison(vib_shap_defect, fusion_shap_defect, title="Head 1: Defect Gate")
    fig_h1.savefig(output_dir / "beeswarm_head1_defect_gate.png", dpi=150, bbox_inches="tight")

    for i, class_name in enumerate(class_names):
        fig_class = plot_beeswarm_comparison(
            vib_shap_fault[:, :, i], fusion_shap_fault[:, :, i], title=f"Head 2: {class_name}")
        fig_class.savefig(output_dir / f"beeswarm_head2_{class_name}.png", dpi=150, bbox_inches="tight")

    logger.info("Plotting dependence plots (top 2 fusion features)...")
    top_features = select_top_features(fusion_shap_defect, k=2)
    for feature_name in top_features:
        fig_dep = plot_dependence(fusion_shap_defect, feature_name, title=f"{feature_name} dependence (fusion)")
        fig_dep.savefig(output_dir / f"dependence_{feature_name.replace(' ', '_')}.png",
                         dpi=150, bbox_inches="tight")

    logger.info("Selecting waterfall case study example...")
    case_study_rows = find_vibration_fails_fusion_succeeds_examples(
        vib_is_defect_pred, fusion_is_defect_pred, test_subset["is_defect"].to_numpy(), n=1)
    if case_study_rows:
        row_idx = case_study_rows[0]
        fig_waterfall = plot_waterfall(fusion_shap_defect, row_idx, title="Vibration wrong, fusion correct")
        fig_waterfall.savefig(output_dir / "waterfall_case_study.png", dpi=150, bbox_inches="tight")
        logger.info("Waterfall case study saved (row %d).", row_idx)
    else:
        logger.warning("No vibration-fails/fusion-succeeds example found in this test subset- "
                        "skipping waterfall case study.")

    logger.info("SHAP analysis complete. Results saved to %s", output_dir)