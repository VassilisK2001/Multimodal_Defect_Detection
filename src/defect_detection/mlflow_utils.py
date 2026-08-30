"""
Loads registered MultimodalDefectClassifier models and their vibration
normalization stats from MLflow. Stats are loaded per run_id, not per
model, since they were logged as a training-run artifact
(vib_normalization_stats.npz) alongside the model, not stored in the
Model Registry itself.

Exported functions:
    get_run_id_for_model_version(modality, version): resolves a
        registered model version to the run that produced it.
    load_registered_model(modality, version, device): loads the model
        itself from the Model Registry.
    load_vib_stats_for_run(run_id): loads (vib_mean, vib_std) from a
        specific run's logged artifact.
    load_model_and_stats(modality, version, device): composes the three functions 
    above into (model, vib_mean, vib_std).
"""


import numpy as np
import mlflow
import mlflow.pytorch as pt
from mlflow import MlflowClient
import torch

from defect_detection.models.fusion_model import MultimodalDefectClassifier


def get_run_id_for_model_version(modality: str, version: str = "latest") -> str:
    """Resolve a registered model version to the run_id that produced it.

    Args:
        modality: "both", "image", or "vibration".
        version: Registered model version, or "latest".

    Returns:
        The run_id of the run that logged this model version.
    """
    registered_name = f"defect_detection_{modality}"
    client = MlflowClient()

    if version == "latest":
        version = client.get_latest_versions(registered_name)[0].version

    run_id = client.get_model_version(registered_name, version).run_id
    assert run_id is not None, (
        f"Model version {version} of '{registered_name}' has no associated run_id."
    )
    return run_id


def load_registered_model(modality: str, version: str = "latest",
                           device: torch.device = torch.device("cpu")) -> MultimodalDefectClassifier:
    """Load a registered MultimodalDefectClassifier from the Model Registry.

    Args:
        modality: "both", "image", or "vibration".
        version: Registered model version, or "latest".
        device: Target device, defaults to  "cpu".

    Returns:
        The loaded model on the requested device.
    """
    registered_name = f"defect_detection_{modality}"
    model_uri = f"models:/{registered_name}/{version}"
    model = pt.load_model(model_uri, map_location=device)

    assert isinstance(model, MultimodalDefectClassifier), (
        f"Expected a MultimodalDefectClassifier at {model_uri}, got {type(model)}"
    )
    return model


def load_vib_stats_for_run(run_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Load the vibration normalization stats logged as an artifact of a given run.

    Args:
        run_id: The MLflow run ID that logged vib_normalization_stats.npz.

    Returns:
        (vib_mean, vib_std).
    """
    client = MlflowClient()
    stats_path = client.download_artifacts(run_id, "vib_normalization_stats.npz")
    stats = np.load(stats_path)
    return stats["mean"], stats["std"]


def load_model_and_stats(modality: str, version: str = "latest", 
                        device: torch.device = torch.device("cpu")) -> tuple[MultimodalDefectClassifier, np.ndarray, np.ndarray]:
    """Load a registered model and the vibration normalization stats logged
    in the same run.

    Args:
        modality: "both", "image", or "vibration".
        version: Registered model version, or "latest".
        device: Target device for the loaded model

    Returns:
        (model, vib_mean, vib_std)
    """
    model = load_registered_model(modality, version, device=device)
    run_id = get_run_id_for_model_version(modality, version)
    vib_mean, vib_std = load_vib_stats_for_run(run_id)
    return model, vib_mean, vib_std