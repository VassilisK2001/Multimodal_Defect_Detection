
import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from defect_detection.mlflow_utils import (
    get_run_id_for_model_version,
    load_model_and_stats,
    load_registered_model,
    load_vib_stats_for_run,
)
from defect_detection.models.fusion_model import MultimodalDefectClassifier


def test_get_run_id_raises_when_run_id_is_none():
    """Should raise if a model version has no associated run_id."""
    fake_version = MagicMock(run_id=None)

    with patch("defect_detection.mlflow_utils.MlflowClient") as mock_client_class:
        mock_client_class.return_value.get_model_version.return_value = fake_version

        with pytest.raises(AssertionError, match="no associated run_id"):
            get_run_id_for_model_version("both", version="1")


def test_get_run_id_returns_value_when_present():
    fake_version = MagicMock(run_id="abc123")

    with patch("defect_detection.mlflow_utils.MlflowClient") as mock_client_class:
        mock_client_class.return_value.get_model_version.return_value = fake_version

        result = get_run_id_for_model_version("both", version="1")

    assert result == "abc123"


def test_get_run_id_resolves_latest_via_get_latest_versions():
    """version='latest' should be resolved to a real version number via
    before calling get_model_version."""
    fake_latest = MagicMock(version="3")
    fake_version = MagicMock(run_id="run_from_latest")

    with patch("defect_detection.mlflow_utils.MlflowClient") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.get_latest_versions.return_value = [fake_latest]
        mock_client.get_model_version.return_value = fake_version

        result = get_run_id_for_model_version("both", version="latest")

    mock_client.get_latest_versions.assert_called_once_with("defect_detection_both")
    mock_client.get_model_version.assert_called_once_with("defect_detection_both", "3")
    assert result == "run_from_latest"


def test_get_run_id_does_not_call_get_latest_versions_for_explicit_version():
    """An explicit version number should be passed through directly, without
    resolving 'latest'."""
    fake_version = MagicMock(run_id="run_2")

    with patch("defect_detection.mlflow_utils.MlflowClient") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.get_model_version.return_value = fake_version

        get_run_id_for_model_version("both", version="2")

    mock_client.get_latest_versions.assert_not_called()
    mock_client.get_model_version.assert_called_once_with("defect_detection_both", "2")


def test_load_registered_model_raises_on_wrong_model_type():
    """Should raise if the loaded artifact isn't a MultimodalDefectClassifier."""
    wrong_model = nn.Linear(5, 3)

    with patch("defect_detection.mlflow_utils.mlflow.pytorch.load_model", return_value=wrong_model):
        with pytest.raises(AssertionError, match="Expected a MultimodalDefectClassifier"):
            load_registered_model("both", version="1")


def test_load_registered_model_returns_model_on_correct_type():
    real_model = MultimodalDefectClassifier(modality="vibration")

    with patch("defect_detection.mlflow_utils.mlflow.pytorch.load_model", return_value=real_model):
        result = load_registered_model("vibration", version="1")

    assert result is real_model


def test_load_registered_model_constructs_correct_uri():
    real_model = MultimodalDefectClassifier(modality="image")

    with patch("defect_detection.mlflow_utils.mlflow.pytorch.load_model", return_value=real_model) as mock_load:
        load_registered_model("image", version="4")

    mock_load.assert_called_once_with("models:/defect_detection_image/4", map_location=torch.device("cpu"))


def test_load_registered_model_uses_requested_device():
    real_model = MultimodalDefectClassifier(modality="image")

    with patch("defect_detection.mlflow_utils.mlflow.pytorch.load_model", return_value=real_model) as mock_load:
        load_registered_model("image", version="4", device=torch.device("cuda"))

    mock_load.assert_called_once_with("models:/defect_detection_image/4", map_location=torch.device("cuda"))



def test_load_vib_stats_parses_real_npz_file():
    """Uses a real .npz file to catch a real serialization mismatch."""
    mean = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    std = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmp_dir:
        npz_path = os.path.join(tmp_dir, "vib_normalization_stats.npz")
        np.savez(npz_path, mean=mean, std=std)

        with patch("defect_detection.mlflow_utils.MlflowClient") as mock_client_class:
            mock_client_class.return_value.download_artifacts.return_value = npz_path

            result_mean, result_std = load_vib_stats_for_run("some_run_id")

    assert np.allclose(result_mean, mean)
    assert np.allclose(result_std, std)


def test_load_model_and_stats_uses_same_run_id_for_both():
    """the run_id used to fetch stats must match the run_id 
    resolved from the model version."""
    real_model = MultimodalDefectClassifier(modality="vibration")
    fake_version = MagicMock(run_id="the_correct_run_id")
    mean = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    std = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)

    with patch("defect_detection.mlflow_utils.mlflow.pytorch.load_model", return_value=real_model), \
         patch("defect_detection.mlflow_utils.MlflowClient") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.get_model_version.return_value = fake_version

        with tempfile.TemporaryDirectory() as tmp_dir:
            npz_path = os.path.join(tmp_dir, "vib_normalization_stats.npz")
            np.savez(npz_path, mean=mean, std=std)
            mock_client.download_artifacts.return_value = npz_path

            load_model_and_stats("vibration", version="1")

    mock_client.download_artifacts.assert_called_once_with(
        "the_correct_run_id", "vib_normalization_stats.npz",
    )


def test_load_model_and_stats_returns_correct_values():
    """The returned (model, mean, std) tuple should contain exactly the values
    produced by the underlying model-loading and stats-loading calls."""
    real_model = MultimodalDefectClassifier(modality="vibration")
    fake_version = MagicMock(run_id="some_run_id")
    mean = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    std = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)

    with patch("defect_detection.mlflow_utils.mlflow.pytorch.load_model", return_value=real_model), \
         patch("defect_detection.mlflow_utils.MlflowClient") as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.get_model_version.return_value = fake_version

        with tempfile.TemporaryDirectory() as tmp_dir:
            npz_path = os.path.join(tmp_dir, "vib_normalization_stats.npz")
            np.savez(npz_path, mean=mean, std=std)
            mock_client.download_artifacts.return_value = npz_path

            result_model, result_mean, result_std = load_model_and_stats("vibration", version="1")

    assert result_model is real_model
    assert np.allclose(result_mean, mean)
    assert np.allclose(result_std, std)

