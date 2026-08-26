import json
import time
from pathlib import Path
from typing import Optional

import requests


def load_manifest(manifest_path: Path) -> dict:
    """Load demo_samples/manifest.json and organize it by scenario.

    Args:
        manifest_path: Path to manifest.json.

    Returns:
        Dict scenario_name -> list of manifest entries (dicts with 'scenario',
        'sample', 'true_is_defect', 'true_fault_class'), in manifest order.
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    scenarios: dict = {}
    for entry in entries:
        scenarios.setdefault(entry["scenario"], []).append(entry)
    return scenarios


def call_inspect_api(image_path: Path, vibration_window_path: Path, api_url: str) -> tuple:
    """Call the /inspect endpoint with a demo sample's files, timing the request.

    Args:
        image_path: Path to the sample's part.png.
        vibration_window_path: Path to the sample's vibration_window.npy.
        api_url: Full URL of the /inspect endpoint.

    Returns:
        (response_json, latency_ms).
    """
    with open(image_path, "rb") as image_file, open(vibration_window_path, "rb") as window_file:
        files = {
            "image": ("part.png", image_file, "image/png"),
            "vibration_window": ("vibration_window.npy", window_file, "application/octet-stream"),
        }
        start = time.perf_counter()
        response = requests.post(api_url, files=files)
        latency_ms = (time.perf_counter() - start) * 1000

    response.raise_for_status()
    return response.json(), latency_ms


def is_prediction_correct(true_is_defect: bool, true_fault_class: Optional[str],
                           predicted_status: str, predicted_fault_type: Optional[str]) -> bool:
    """Compare a prediction against ground truth.

    Args:
        true_is_defect: Ground-truth defect status.
        true_fault_class: Ground-truth fault class, or None for normal samples.
        predicted_status: 'healthy' or 'defective', from the API response.
        predicted_fault_type: Predicted fault class, or None.

    Returns:
        True if the prediction matches ground truth (defect status, and fault
        type too when the sample is actually defective).
    """
    predicted_is_defect = predicted_status == "defective"
    if predicted_is_defect != true_is_defect:
        return False
    if true_is_defect:
        return predicted_fault_type == true_fault_class
    return True