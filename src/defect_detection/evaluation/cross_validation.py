
from typing import cast
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from defect_detection.data.dataset import MultimodalDefectDataset
from defect_detection.data.splitting import generate_stratified_kfold_splits
from defect_detection.evaluation.metrics import compute_defect_gate_metrics, compute_fault_type_metrics
from defect_detection.evaluation.predictions import collect_test_predictions
from defect_detection.training.train import train_from_dataframes
from defect_detection.models.fusion_model import Modality


def run_kfold_cv(modality: Modality, manifest_df: pd.DataFrame, window_size: int, fs: int, class_names: list[str],
                  k: int = 3, seed: int = 42, device: torch.device = torch.device("cpu")) -> list[dict]:
    """Run k-fold cross-validation for one modality.

    Args:
        modality: "both", "image", or "vibration".
        manifest_df: The full manifest (all rows, unsplit).
        window_size: Vibration window size in samples.
        fs: Vibration sampling rate in Hz.
        class_names: Fault class names, in index order.
        k: Number of folds.
        seed: Base random seed for fold generation and training.
        device: Device to train and evaluate on.

    Returns:
        A list of k dicts, each with 'defect_metrics' and 'fault_metrics'.
    """
    folds = generate_stratified_kfold_splits(manifest_df, k=k, seed=seed)
    fold_results = []

    for fold_idx, fold_df in enumerate(folds):
        train_df = cast(pd.DataFrame, fold_df[fold_df.split == "train"])
        val_df = cast(pd.DataFrame, fold_df[fold_df.split == "val"])
        test_df = cast(pd.DataFrame, fold_df[fold_df.split == "test"]) 

        model, vib_mean, vib_std = train_from_dataframes(
            train_df, val_df, modality=modality, seed=seed + fold_idx,
            run_name_suffix=f"_cv_fold{fold_idx}", register_model=False,
        )

        test_dataset = MultimodalDefectDataset(
            test_df, window_size=window_size, fs=fs, training=False,
            vib_mean=vib_mean, vib_std=vib_std,
        )
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        predictions = collect_test_predictions(model, test_loader, device=device)

        fold_results.append({
            "defect_metrics": compute_defect_gate_metrics(
                predictions["is_defect_true"], predictions["is_defect_pred"],
            ),
            "fault_metrics": compute_fault_type_metrics(
                predictions["fault_class_true"], predictions["fault_class_pred"], class_names,
            ),
        })

    return fold_results

def _aggregate_leaf(values: list) -> dict:
    return {"mean": float(np.mean(values)), "std": float(np.std(values))}


def aggregate_cv_results(fold_results: list[dict]) -> dict:
    """Aggregate mean/std for every metric across folds, preserving structure.

    Args:
        fold_results: A list of per-fold result dicts, all sharing the same
            nested structure.

    Returns:
        The same nested structure, with each numeric leaf replaced by
        {"mean": ..., "std": ...} computed across folds.
    """
    def _recurse(nodes: list):
        if isinstance(nodes[0], dict):
            return {key: _recurse([n[key] for n in nodes]) for key in nodes[0]}
        return _aggregate_leaf(nodes)

    return _recurse(fold_results)