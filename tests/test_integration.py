
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from defect_detection.data.dataset import MultimodalDefectDataset
from defect_detection.models.fusion_model import MultimodalDefectClassifier
from defect_detection.utils import find_project_root, load_yaml_config


@pytest.fixture(scope="module")
def config() -> dict:
    return load_yaml_config("config/data_config.yaml")


@pytest.fixture(scope="module")
def manifest_df(config) -> pd.DataFrame:
    project_root = find_project_root()
    manifest_path = project_root / config["paths"]["manifest_dir"] / "manifest.csv"
    return pd.read_csv(manifest_path)


@pytest.fixture(scope="module")
def real_batch(manifest_df, config):
    """A real batch from the data pipeline (small subset, for speed)."""
    small_df = manifest_df.head(8)
    dataset = MultimodalDefectDataset(
        small_df, window_size=config["window_size"],
        fs=config["cwru"]["sampling_rate_hz"], training=False,
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    return next(iter(loader))


@pytest.mark.parametrize("modality", ["both", "image", "vibration"])
def test_real_batch_forward_pass_no_errors(real_batch, modality):
    """A real batch should run through each model variant without shape errors."""
    images, vib_features, is_defect, fault_class_idx, area_ratio = real_batch
    model = MultimodalDefectClassifier(modality=modality)

    kwargs = {}
    if modality in ("both", "image"):
        kwargs["image"] = images
    if modality in ("both", "vibration"):
        kwargs["vib_features"] = vib_features

    defect_logit, fault_logits = model(**kwargs)

    assert defect_logit.shape == (images.shape[0], 1)
    assert fault_logits.shape == (images.shape[0], 3)


def test_real_batch_labels_have_expected_dtypes(real_batch):
    """Labels should be float32 (is_defect) and long (fault_class_idx)."""
    _, _, is_defect, fault_class_idx, _ = real_batch
    assert is_defect.dtype == torch.float32
    assert fault_class_idx.dtype == torch.long


@pytest.mark.parametrize("modality", ["both", "image", "vibration"])
def test_real_batch_loss_computation_runs(real_batch, modality):
    """A forward, loss, and backward pass on a real batch should run without error."""
    images, vib_features, is_defect, fault_class_idx, _ = real_batch
    model = MultimodalDefectClassifier(modality=modality)

    kwargs = {}
    if modality in ("both", "image"):
        kwargs["image"] = images
    if modality in ("both", "vibration"):
        kwargs["vib_features"] = vib_features

    defect_logit, fault_logits = model(**kwargs)

    defect_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        defect_logit.squeeze(1), is_defect,
    )

    defect_mask = is_defect.bool()
    if defect_mask.sum() > 0:
        fault_loss = torch.nn.functional.cross_entropy(
            fault_logits[defect_mask], fault_class_idx[defect_mask],
        )
        total_loss = defect_loss + fault_loss
    else:
        total_loss = defect_loss

    total_loss.backward()
    assert total_loss.item() > 0