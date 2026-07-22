
import argparse
import tempfile
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
from mlflow.models import infer_signature
from torch.utils.data import DataLoader

from defect_detection.data.dataset import MultimodalDefectDataset
from defect_detection.data.normalization import compute_vibration_feature_stats
from defect_detection.models.fusion_model import MultimodalDefectClassifier
from defect_detection.training.losses import (
    TwoStageLoss,
    compute_defect_gate_pos_weight,
    compute_fault_type_class_weights,
)
from defect_detection.training.visualization import (
    plot_defect_accuracy_curve,
    plot_defect_loss_curve,
    plot_fault_type_accuracy_curve,
    plot_fault_type_loss_curve,
)
from defect_detection.utils import find_project_root, flatten_dict, load_yaml_config


def load_split_dataframes(project_root: Path, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/val/test manifest CSVs.

    Args:
        project_root: Project root path.
        config: data_config.yaml contents.

    Returns:
        (train_df, val_df, test_df).
    """
    manifest_dir = project_root / config["paths"]["manifest_dir"]
    train_df = pd.read_csv(manifest_dir / "train.csv")
    val_df = pd.read_csv(manifest_dir / "val.csv")
    test_df = pd.read_csv(manifest_dir / "test.csv")
    return train_df, val_df, test_df


def build_datasets(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                    window_size: int, fs: int
                    ) -> tuple[MultimodalDefectDataset, MultimodalDefectDataset,
                               MultimodalDefectDataset, np.ndarray, np.ndarray]:
    """Build train/val/test Datasets, with vibration features normalized using stats
    computed from the training split.

    Args:
        train_df: Training split manifest.
        val_df: Validation split manifest.
        test_df: Test split manifest.
        window_size: Vibration window size in samples.
        fs: Vibration sampling rate in Hz.

    Returns:
        (train_dataset, val_dataset, test_dataset, vib_mean, vib_std).
    """
    raw_train_dataset = MultimodalDefectDataset(train_df, window_size=window_size, fs=fs, training=False)
    vib_mean, vib_std = compute_vibration_feature_stats(raw_train_dataset)

    train_dataset = MultimodalDefectDataset(
        train_df, window_size=window_size, fs=fs, training=True, vib_mean=vib_mean, vib_std=vib_std,
    )
    val_dataset = MultimodalDefectDataset(
        val_df, window_size=window_size, fs=fs, training=False, vib_mean=vib_mean, vib_std=vib_std,
    )
    test_dataset = MultimodalDefectDataset(
        test_df, window_size=window_size, fs=fs, training=False, vib_mean=vib_mean, vib_std=vib_std,
    )
    return train_dataset, val_dataset, test_dataset, vib_mean, vib_std


def build_optimizer(model: MultimodalDefectClassifier, train_config: dict) -> torch.optim.Optimizer:
    """Build an AdamW optimizer with a reduced learning rate for the pretrained
    image encoder layers.

    Args:
        model: The model to optimize.
        train_config: train_config.yaml contents.

    Returns:
        A configured AdamW optimizer.
    """
    base_lr = train_config["optimizer"]["lr"]
    weight_decay = train_config["optimizer"]["weight_decay"]
    fine_tune_lr = base_lr * train_config["optimizer"]["fine_tune_lr_multiplier"]

    param_groups = []
    if model.image_encoder is not None:
        param_groups.append({"params": model.image_encoder.parameters(), "lr": fine_tune_lr})
    if model.vibration_encoder is not None:
        param_groups.append({"params": model.vibration_encoder.parameters(), "lr": base_lr})
    param_groups.append({"params": model.fusion_mlp.parameters(), "lr": base_lr})
    param_groups.append({"params": model.defect_head.parameters(), "lr": base_lr})
    param_groups.append({"params": model.fault_type_head.parameters(), "lr": base_lr})

    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def _forward_batch(model: MultimodalDefectClassifier, batch, device: torch.device):
    """Move a batch to device and run the model's forward pass."""
    images, vib_features, is_defect, fault_class_idx, _ = batch
    is_defect = is_defect.to(device)
    fault_class_idx = fault_class_idx.to(device)

    kwargs = {}
    if model.image_encoder is not None:
        kwargs["image"] = images.to(device)
    if model.vibration_encoder is not None:
        kwargs["vib_features"] = vib_features.to(device)

    defect_logit, fault_type_logits = model(**kwargs)
    return defect_logit, fault_type_logits, is_defect, fault_class_idx


def train_one_epoch(model: MultimodalDefectClassifier, loader: DataLoader,
                     criterion: TwoStageLoss, optimizer: torch.optim.Optimizer,
                     device: torch.device) -> dict:
    """Run one training epoch.

    Args:
        model: The model to train.
        loader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to run on.

    Returns:
        Dict with 'defect_loss', 'fault_type_loss', 'defect_accuracy',
        'fault_type_accuracy'. fault_type_loss/accuracy are NaN if no defective
        samples were seen during the epoch.
    """
    model.train()
    total_defect_loss, total_fault_loss, total_defective_seen = 0.0, 0.0, 0
    n_batches = 0
    correct_defect, total_samples = 0, 0
    correct_fault_type = 0

    for batch in loader:
        defect_logit, fault_type_logits, is_defect, fault_class_idx = _forward_batch(model, batch, device)

        optimizer.zero_grad()
        total_loss, defect_loss, fault_type_loss, n_defective = criterion(
            defect_logit, fault_type_logits, is_defect, fault_class_idx,
        )
        total_loss.backward()
        optimizer.step()

        total_defect_loss += defect_loss.item()
        n_batches += 1
        if fault_type_loss is not None:
            total_fault_loss += fault_type_loss.item() * n_defective
            total_defective_seen += n_defective

        with torch.no_grad():
            preds = (torch.sigmoid(defect_logit.squeeze(1)) >= 0.5).float()
            correct_defect += (preds == is_defect).sum().item()
            total_samples += is_defect.size(0)

            if n_defective > 0:
                defect_mask = is_defect.bool()
                fault_preds = fault_type_logits[defect_mask].argmax(dim=1)
                correct_fault_type += (fault_preds == fault_class_idx[defect_mask]).sum().item()

    return {
        "defect_loss": total_defect_loss / n_batches,
        "fault_type_loss": (total_fault_loss / total_defective_seen) if total_defective_seen > 0 else float("nan"),
        "defect_accuracy": correct_defect / total_samples,
        "fault_type_accuracy": (correct_fault_type / total_defective_seen) if total_defective_seen > 0 else float("nan"),
    }


@torch.no_grad()
def evaluate(model: MultimodalDefectClassifier, loader: DataLoader,
             criterion: TwoStageLoss, device: torch.device) -> dict:
    """Evaluate the model on a DataLoader without updating its weights.

    Args:
        model: The model to evaluate.
        loader: Evaluation DataLoader.
        criterion: Loss function.
        device: Device to run on.

    Returns:
        Dict with 'defect_loss', 'fault_type_loss', 'defect_accuracy',
        'fault_type_accuracy'. fault_type_loss/accuracy are NaN if no defective
        samples were seen.
    """
    model.eval()
    total_defect_loss, total_fault_loss, total_defective_seen = 0.0, 0.0, 0
    n_batches = 0
    correct_defect, total_samples = 0, 0
    correct_fault_type = 0

    for batch in loader:
        defect_logit, fault_type_logits, is_defect, fault_class_idx = _forward_batch(model, batch, device)

        _, defect_loss, fault_type_loss, n_defective = criterion(
            defect_logit, fault_type_logits, is_defect, fault_class_idx,
        )

        total_defect_loss += defect_loss.item()
        n_batches += 1
        if fault_type_loss is not None:
            total_fault_loss += fault_type_loss.item() * n_defective
            total_defective_seen += n_defective

        preds = (torch.sigmoid(defect_logit.squeeze(1)) >= 0.5).float()
        correct_defect += (preds == is_defect).sum().item()
        total_samples += is_defect.size(0)

        if n_defective > 0:
            defect_mask = is_defect.bool()
            fault_preds = fault_type_logits[defect_mask].argmax(dim=1)
            correct_fault_type += (fault_preds == fault_class_idx[defect_mask]).sum().item()

    return {
        "defect_loss": total_defect_loss / n_batches,
        "fault_type_loss": (total_fault_loss / total_defective_seen) if total_defective_seen > 0 else float("nan"),
        "defect_accuracy": correct_defect / total_samples,
        "fault_type_accuracy": (correct_fault_type / total_defective_seen) if total_defective_seen > 0 else float("nan"),
    }


def train(modality: str = "both", experiment_name: str = "defect_detection"):
    """Train a MultimodalDefectClassifier and log the run to MLflow.

    Args:
        modality: "both", "image", or "vibration" — which encoder(s) feed the model.
        experiment_name: MLflow experiment name.

    Returns:
        The trained model (best early-stopped weights loaded).
    """
    project_root = find_project_root()
    data_config = load_yaml_config("config/data_config.yaml")
    model_config = load_yaml_config("config/model_config.yaml")
    train_config = load_yaml_config("config/train_config.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df, val_df, test_df = load_split_dataframes(project_root, data_config)

    train_dataset, val_dataset, test_dataset, vib_mean, vib_std = build_datasets(
        train_df, val_df, test_df,
        window_size=data_config["window_size"], fs=data_config["cwru"]["sampling_rate_hz"],
    )

    batch_size = train_config["training"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    pos_weight = compute_defect_gate_pos_weight(train_df).to(device)
    fault_type_weights = compute_fault_type_class_weights(train_df).to(device)
    criterion = TwoStageLoss(pos_weight, fault_type_weights)

    model = MultimodalDefectClassifier(modality=modality).to(device)
    optimizer = build_optimizer(model, train_config)

    max_epochs = train_config["training"]["max_epochs"]
    patience = train_config["training"]["early_stopping_patience"]
    best_val_loss = float("inf")
    patience_counter = 0
    history = {
        "train_defect_loss": [], "train_fault_type_loss": [],
        "train_defect_accuracy": [], "train_fault_type_accuracy": [],
        "val_defect_loss": [], "val_fault_type_loss": [],
        "val_defect_accuracy": [], "val_fault_type_accuracy": [],
    }

    mlflow.set_tracking_uri(f"sqlite:///{project_root / 'mlflow.db'}")
    mlflow.set_experiment(experiment_name)

    with tempfile.TemporaryDirectory() as tmp_dir:
        best_checkpoint_path = Path(tmp_dir) / "best_model.pt"
        vib_stats_path = Path(tmp_dir) / "vib_normalization_stats.npz"
        np.savez(vib_stats_path, mean=vib_mean, std=vib_std)

        with mlflow.start_run(run_name=f"{modality}") as run:
            mlflow.log_param("modality", modality)

            flat_params = {}
            flat_params.update({f"model.{k}": v for k, v in flatten_dict(model_config).items()})
            flat_params.update({f"train.{k}": v for k, v in flatten_dict(train_config).items()})
            mlflow.log_params(flat_params)

            mlflow.log_dict(data_config, "config_snapshots/data_config.yaml")
            mlflow.log_dict(model_config, "config_snapshots/model_config.yaml")
            mlflow.log_dict(train_config, "config_snapshots/train_config.yaml")

            mlflow.log_artifact(str(vib_stats_path))

            for epoch in range(max_epochs):
                train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
                val_metrics = evaluate(model, val_loader, criterion, device)

                mlflow.log_metrics({
                    "train_defect_loss": train_metrics["defect_loss"],
                    "train_fault_type_loss": train_metrics["fault_type_loss"],
                    "train_defect_accuracy": train_metrics["defect_accuracy"],
                    "train_fault_type_accuracy": train_metrics["fault_type_accuracy"],
                    "val_defect_loss": val_metrics["defect_loss"],
                    "val_fault_type_loss": val_metrics["fault_type_loss"],
                    "val_defect_accuracy": val_metrics["defect_accuracy"],
                    "val_fault_type_accuracy": val_metrics["fault_type_accuracy"],
                }, step=epoch)

                history["train_defect_loss"].append(train_metrics["defect_loss"])
                history["train_fault_type_loss"].append(train_metrics["fault_type_loss"])
                history["train_defect_accuracy"].append(train_metrics["defect_accuracy"])
                history["train_fault_type_accuracy"].append(train_metrics["fault_type_accuracy"])
                history["val_defect_loss"].append(val_metrics["defect_loss"])
                history["val_fault_type_loss"].append(val_metrics["fault_type_loss"])
                history["val_defect_accuracy"].append(val_metrics["defect_accuracy"])
                history["val_fault_type_accuracy"].append(val_metrics["fault_type_accuracy"])

                val_total_loss = val_metrics["defect_loss"] + (
                    val_metrics["fault_type_loss"] if not np.isnan(val_metrics["fault_type_loss"]) else 0.0
                )

                print(f"Epoch {epoch+1}/{max_epochs} | "
                      f"train_defect={train_metrics['defect_loss']:.4f} "
                      f"train_fault={train_metrics['fault_type_loss']:.4f} | "
                      f"val_defect={val_metrics['defect_loss']:.4f} "
                      f"val_fault={val_metrics['fault_type_loss']:.4f} "
                      f"val_acc={val_metrics['defect_accuracy']:.4f}")

                if val_total_loss < best_val_loss:
                    best_val_loss = val_total_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), best_checkpoint_path)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break

            mlflow.log_artifact(str(best_checkpoint_path))

            mlflow.log_figure(plot_defect_loss_curve(history, modality=modality), "defect_loss_curve.png")
            mlflow.log_figure(plot_fault_type_loss_curve(history, modality=modality), "fault_type_loss_curve.png")
            mlflow.log_figure(plot_defect_accuracy_curve(history, modality=modality), "defect_accuracy_curve.png")
            mlflow.log_figure(
                plot_fault_type_accuracy_curve(history, modality=modality), "fault_type_accuracy_curve.png",
            )

            model.load_state_dict(torch.load(best_checkpoint_path))
            model.eval()

            sample_images, sample_vib, _, _, _ = next(iter(val_loader))
            sample_kwargs = {}
            if model.image_encoder is not None:
                sample_kwargs["image"] = sample_images.to(device)
            if model.vibration_encoder is not None:
                sample_kwargs["vib_features"] = sample_vib.to(device)

            with torch.no_grad():
                sample_output = model(**sample_kwargs)

            input_example = {k: v.cpu().numpy() for k, v in sample_kwargs.items()}
            output_example = tuple(o.cpu().numpy() for o in sample_output)
            signature = infer_signature(input_example, output_example)

            mlflow.pytorch.log_model(
                model, artifact_path="model", signature=signature, input_example=input_example,
            )

            registered_name = f"defect_detection_{modality}"
            model_uri = f"runs:/{run.info.run_id}/model"
            mlflow.register_model(model_uri, registered_name)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=["both", "image", "vibration"], default="both")
    args = parser.parse_args()

    train(modality=args.modality)