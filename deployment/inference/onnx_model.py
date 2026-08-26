from pathlib import Path

import numpy as np
import onnxruntime as ort


class OnnxFusionModel:
    """Loads the exported fusion model once, and exposes a single predict()
    method for repeated per-request use."""

    def __init__(self, onnx_path: Path, intra_op_num_threads: int = 2):
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = intra_op_num_threads
        self.session = ort.InferenceSession(
            str(onnx_path), sess_options=session_options, providers=["CPUExecutionProvider"],
        )

    def predict(self, image: np.ndarray, vib_features: np.ndarray) -> tuple:
        """Run one inference call.

        Args:
            image: (1, 3, 224, 224) preprocessed image array.
            vib_features: (1, 5) preprocessed (normalized) vibration features.

        Returns:
            (defect_proba, fault_proba): a float and a (3,) array.
        """
        defect_proba, fault_proba = self.session.run(
            ["defect_proba", "fault_proba"], {"image": image, "vib_features": vib_features},
        )
        return float(defect_proba[0, 0]), fault_proba[0]