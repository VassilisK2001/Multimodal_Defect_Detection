import logging
from typing import cast

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader

from defect_detection.data.dataset import MultimodalDefectDataset
from defect_detection.data.splitting import generate_stratified_kfold_splits
from defect_detection.evaluation.predictions import collect_test_predictions
from defect_detection.models.fusion_model import Modality
from defect_detection.training.train import train_from_dataframes


logger = logging.getLogger(__name__)

MODALITIES: tuple[Modality, ...] = ("image", "vibration", "both")


def initialize_oof_arrays(n_samples: int, num_fault_classes: int,
                           modalities: tuple[Modality, ...] = MODALITIES) -> dict:
    """Initialize zeroed out-of-fold probability arrays for each model.

    Args:
        n_samples: Total number of rows in the manifest.
        num_fault_classes: Number of fault-type classes (Head 2's output size).
        modalities: Model names to initialize arrays for.

    Returns:
        Dict mapping modality -> {"oof_defect_proba": (n_samples,) zeros,
        "oof_fault_proba": (n_samples, num_fault_classes) zeros}.
    """
    return {
        modality: {
            "oof_defect_proba": np.zeros(n_samples, dtype=float),
            "oof_fault_proba": np.zeros((n_samples, num_fault_classes), dtype=float),
        }
        for modality in modalities
    }


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """roc_auc_score, returning NaN instead of raising when only one class is
    present in y_true."""
    if len(np.unique(y_true)) < 2:
        logger.warning("ROC AUC undefined (only one class present in y_true, n=%d) — returning NaN",
                        len(y_true))
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """average_precision_score, returning NaN when y_true has no positive examples."""
    if y_true.sum() == 0:
        logger.warning("PR AUC undefined (no positive examples in y_true, n=%d) — returning NaN",
                        len(y_true))
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def _compute_fold_fault_auc(y_true: np.ndarray, y_score: np.ndarray,
                             class_names: list[str]) -> dict:
    """One-vs-rest ROC AUC / PR AUC per fault class, for one fold's defective
    validation rows.

    Args:
        y_true: (N,) integer class-index array.
        y_score: (N, len(class_names)) predicted probabilities per class.
        class_names: Class names, in index order.

    Returns:
        Dict mapping each class name to {'roc_auc', 'pr_auc'}. All NaN if the
        fold has zero defective rows. A class's values are NaN if that class
        cannot be evaluated (absent, or present in every row).
    """
    if len(y_true) == 0:
        return {class_name: {"roc_auc": float("nan"), "pr_auc": float("nan")} for class_name in class_names}

    y_true_binarized = label_binarize(y_true, classes=range(len(class_names)))
    return {
        class_name: {
            "roc_auc": _safe_roc_auc(y_true_binarized[:, i], y_score[:, i]),
            "pr_auc": _safe_pr_auc(y_true_binarized[:, i], y_score[:, i]),
        }
        for i, class_name in enumerate(class_names)
    }


def run_oof_cross_validation(manifest_df: pd.DataFrame, class_names: list[str],
                              window_size: int, fs: int, k: int = 3, seed: int = 42,
                              device: torch.device = torch.device("cpu"),
                              modalities: tuple[Modality, ...] = MODALITIES) -> dict:
    """Run stratified k-fold cross-validation for all modalities on one shared
    fold split, collecting out-of-fold (OOF) predictions and per-fold metrics.

    Args:
        manifest_df: The full manifest (all rows, unsplit). Must have a standard
            0..N-1 RangeIndex, since row positions are used to place predictions into the OOF arrays.
        class_names: Fault class names, in index order.
        window_size: Vibration window size in samples.
        fs: Vibration sampling rate in Hz.
        k: Number of folds.
        seed: Base random seed for fold generation and training.
        device: Device to train and evaluate on.
        modalities: Which models to run.

    Returns:
        Dict with:
            "is_defect_true": (N,) int array, ground truth for Head 1.
            "fault_class_true": (N,) int array, ground truth fault class index for
                defective rows; -1 for normal rows (not a valid class index).
            "models": modality -> {"oof_defect_proba", "oof_fault_proba",
                "fold_metrics"}.
    """
    n_samples = len(manifest_df)
    oof = initialize_oof_arrays(n_samples, len(class_names), modalities)
    is_defect_true = np.zeros(n_samples, dtype=int)
    fault_class_true = np.full(n_samples, -1, dtype=int)
    for modality in modalities:
        oof[modality]["fold_metrics"] = []

    fault_class_to_idx = {name: i for i, name in enumerate(class_names)}
    folds = generate_stratified_kfold_splits(manifest_df, k=k, seed=seed)
    logger.info("Starting OOF cross-validation: k=%d folds, modalities=%s", k, modalities)

    for fold_idx, fold_df in enumerate(folds):
        logger.info("--- Fold %d/%d ---", fold_idx + 1, k)
        train_df = cast(pd.DataFrame, fold_df[fold_df.split == "train"])
        val_df = cast(pd.DataFrame, fold_df[fold_df.split == "val"])
        test_df = cast(pd.DataFrame, fold_df[fold_df.split == "test"])

        global_positions = test_df.index.to_numpy()
        test_df_reset = test_df.reset_index(drop=True)

        is_defect_true[global_positions] = test_df_reset["is_defect"].to_numpy()
        defective_mask = test_df_reset["is_defect"].to_numpy() == 1
        fault_class_true[global_positions[defective_mask]] = (
            test_df_reset.loc[defective_mask, "fault_class"].map(fault_class_to_idx).to_numpy()
        )

        for modality in modalities:
            logger.info("[fold %d/%d] Training modality '%s' (train=%d, val=%d, test=%d rows)...",
                        fold_idx + 1, k, modality, len(train_df), len(val_df), len(test_df))

            model, vib_mean, vib_std = train_from_dataframes(
                train_df, val_df, modality=modality, seed=seed + fold_idx,
                run_name_suffix=f"_oof_fold{fold_idx}", register_model=False,
            )

            test_dataset = MultimodalDefectDataset(
                test_df_reset, window_size=window_size, fs=fs, training=False,
                vib_mean=vib_mean, vib_std=vib_std,
            )
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            predictions = collect_test_predictions(model, test_loader, device=device)

            oof[modality]["oof_defect_proba"][global_positions] = predictions["defect_proba"]
            defective_positions = global_positions[defective_mask]
            oof[modality]["oof_fault_proba"][defective_positions] = predictions["fault_class_proba"]

            defect_roc_auc = _safe_roc_auc(predictions["is_defect_true"], predictions["defect_proba"])
            defect_pr_auc = _safe_pr_auc(predictions["is_defect_true"], predictions["defect_proba"])

            oof[modality]["fold_metrics"].append({
                "defect": {"roc_auc": defect_roc_auc, "pr_auc": defect_pr_auc},
                "fault": _compute_fold_fault_auc(
                    predictions["fault_class_true"], predictions["fault_class_proba"], class_names,
                ),
            })

            logger.info("[fold %d/%d] '%s' complete — defect ROC AUC=%.3f, PR AUC=%.3f",
                        fold_idx + 1, k, modality, defect_roc_auc, defect_pr_auc)

    logger.info("OOF cross-validation complete: %d folds x %d modalities.", k, len(modalities))

    return {
        "is_defect_true": is_defect_true,
        "fault_class_true": fault_class_true,
        "models": {m: oof[m] for m in modalities},
    }


def print_fold_level_summary(oof_results: dict, class_names: list[str]) -> None:
    """Print mean ± std across folds for Head 1 and Head 2 (per class), per model."""
    for modality, model_data in oof_results["models"].items():
        n_folds = len(model_data["fold_metrics"])
        print(f"\n=== {modality} — fold-level (mean \u00b1 std across {n_folds} folds) ===")

        defect_roc = [f["defect"]["roc_auc"] for f in model_data["fold_metrics"]]
        defect_pr = [f["defect"]["pr_auc"] for f in model_data["fold_metrics"]]
        print(f"Head 1 (defect gate):  "
              f"ROC AUC = {np.nanmean(defect_roc):.3f} \u00b1 {np.nanstd(defect_roc):.3f}   "
              f"PR AUC = {np.nanmean(defect_pr):.3f} \u00b1 {np.nanstd(defect_pr):.3f}")

        print("Head 2 (fault type):")
        for class_name in class_names:
            roc_vals = [f["fault"][class_name]["roc_auc"] for f in model_data["fold_metrics"]]
            pr_vals = [f["fault"][class_name]["pr_auc"] for f in model_data["fold_metrics"]]
            print(f"  {class_name:12} "
                  f"ROC AUC = {np.nanmean(roc_vals):.3f} \u00b1 {np.nanstd(roc_vals):.3f}   "
                  f"PR AUC = {np.nanmean(pr_vals):.3f} \u00b1 {np.nanstd(pr_vals):.3f}")


def compute_global_oof_metrics(oof_results: dict, class_names: list[str]) -> dict:
    """Compute global out-of-fold ROC AUC / PR AUC for Head 1 (all samples) and
    Head 2 (defective samples only, one-vs-rest per class), per model.

    Args:
        oof_results: Output of run_oof_cross_validation().
        class_names: Fault class names, in index order.

    Returns:
        Dict: modality -> {"defect": {"roc_auc", "pr_auc"},
                            "fault": {class_name: {"roc_auc", "pr_auc"}, ...}}.
    """
    is_defect_true = oof_results["is_defect_true"]
    fault_class_true = oof_results["fault_class_true"]
    defective_mask = is_defect_true == 1
    fault_true_binarized = label_binarize(
        fault_class_true[defective_mask], classes=range(len(class_names)),
    )

    global_metrics = {}
    for modality, model_data in oof_results["models"].items():
        defect_proba = model_data["oof_defect_proba"]
        fault_proba_defective = model_data["oof_fault_proba"][defective_mask]

        global_metrics[modality] = {
            "defect": {
                "roc_auc": _safe_roc_auc(is_defect_true, defect_proba),
                "pr_auc": _safe_pr_auc(is_defect_true, defect_proba),
            },
            "fault": {
                class_name: {
                    "roc_auc": _safe_roc_auc(fault_true_binarized[:, i], fault_proba_defective[:, i]),
                    "pr_auc": _safe_pr_auc(fault_true_binarized[:, i], fault_proba_defective[:, i]),
                }
                for i, class_name in enumerate(class_names)
            },
        }

    return global_metrics


def print_global_oof_metrics(global_metrics: dict, class_names: list[str]) -> None:
    """Print the global OOF ROC AUC / PR AUC per model, Head 1 and Head 2."""
    for modality, metrics in global_metrics.items():
        print(f"\n=== {modality} — global OOF ===")
        print(f"Head 1 (defect gate):  "
              f"ROC AUC = {metrics['defect']['roc_auc']:.3f}   "
              f"PR AUC = {metrics['defect']['pr_auc']:.3f}")
        print("Head 2 (fault type):")
        for class_name in class_names:
            m = metrics["fault"][class_name]
            print(f"  {class_name:12} ROC AUC = {m['roc_auc']:.3f}   PR AUC = {m['pr_auc']:.3f}")