
import numpy as np
import torch
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset

from defect_detection.data.augmentations import build_image_transform, jitter, scale
from defect_detection.data.features import extract_features
from defect_detection.data.normalization import apply_vibration_normalization
from defect_detection.utils import find_project_root


class MultimodalDefectDataset(Dataset):
    """
    Each row of the manifest yields:
        image           - (3, 224, 224) normalized tensor
        vib_features    - (5,) float32 tensor: RMS, Peak, Crest Factor, Spectral Kurtosis, TKEO
        is_defect       - scalar float32 (0.0 / 1.0), target for the binary defect gate
        fault_class_idx - scalar long (-1 / 0 / 1 / 2), target for the fault-type head
                          (-1 is a sentinel for normal samples, masked out in the loss)
        area_ratio      - scalar float32, defect_area_ratio (0.0 for normal samples),
                          kept for potential later use (e.g. weighted sampling)
    """

    def __init__(self, manifest_df, window_size: int, fs: int, training: bool = False,
                 vib_mean: np.ndarray | None = None, vib_std: np.ndarray | None = None):
        self.df = manifest_df.reset_index(drop=True)
        self.window_size = window_size
        self.fs = fs
        self.training = training
        self.project_root = find_project_root()
        self.image_transform = build_image_transform(training)
        self._mat_cache: dict[str, np.ndarray] = {}

        # Optional vibration normalization stats, computed from the training set.
        self.vib_mean = vib_mean
        self.vib_std = vib_std

    def __len__(self) -> int:
        return len(self.df)

    def _load_de_signal(self, relative_mat_path: str) -> np.ndarray:
        if relative_mat_path not in self._mat_cache:
            mat_path = self.project_root / relative_mat_path
            mat = loadmat(mat_path)
            de_key = [k for k in mat.keys() if "DE_time" in k][0]
            self._mat_cache[relative_mat_path] = mat[de_key].flatten()
        return self._mat_cache[relative_mat_path]

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # Image branch 
        image_path = self.project_root / row.image_path
        image = Image.open(image_path).convert("RGB")
        image = self.image_transform(image)

        # Vibration branch 
        signal = self._load_de_signal(row.vibration_file)
        start = row.vibration_window_idx * self.window_size
        window = signal[start:start + self.window_size].astype(np.float32)

        if self.training:
            window = jitter(window)
            window = scale(window)

        vib_features = extract_features(window, fs=self.fs)
        if self.vib_mean is not None and self.vib_std is not None:
            vib_features = apply_vibration_normalization(vib_features, self.vib_mean, self.vib_std)
        vib_tensor = torch.tensor(vib_features, dtype=torch.float32)

        # Labels
        is_defect = torch.tensor(row.is_defect, dtype=torch.float32)
        fault_class_idx = torch.tensor(row.fault_class_idx, dtype=torch.long)
        area_ratio = torch.tensor(row.defect_area_ratio, dtype=torch.float32)

        return image, vib_tensor, is_defect, fault_class_idx, area_ratio