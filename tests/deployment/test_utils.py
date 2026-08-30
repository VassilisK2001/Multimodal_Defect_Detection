"""
Tests for src/defect_detection/deployment/utils.py.
"""


import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

DEPLOYMENT_DIR = Path(__file__).resolve().parents[2] / "deployment"
sys.path.insert(0, str(DEPLOYMENT_DIR))


from utils import call_inspect_api, is_prediction_correct, load_manifest


def test_load_manifest_groups_entries_by_scenario(tmp_path):
    entries = [
        {"scenario": "normal", "sample": "sample_1", "true_is_defect": False, "true_fault_class": None},
        {"scenario": "normal", "sample": "sample_2", "true_is_defect": False, "true_fault_class": None},
        {"scenario": "outer_race", "sample": "sample_1", "true_is_defect": True, "true_fault_class": "outer_race"},
    ]
    manifest_path = tmp_path / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(entries, f)

    result = load_manifest(manifest_path)

    assert set(result.keys()) == {"normal", "outer_race"}
    assert len(result["normal"]) == 2
    assert len(result["outer_race"]) == 1
    assert result["normal"][0]["sample"] == "sample_1"


def test_load_manifest_empty_file(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump([], f)

    result = load_manifest(manifest_path)

    assert result == {}


def test_call_inspect_api_returns_response_json_and_positive_latency(tmp_path):
    image_path = tmp_path / "part.png"
    Image.new("RGB", (10, 10)).save(image_path)
    window_path = tmp_path / "vibration_window.npy"
    np.save(window_path, np.zeros(2048, dtype=np.float32))

    fake_response = MagicMock()
    fake_response.json.return_value = {"status": "healthy", "defect_probability": 0.1}
    fake_response.raise_for_status.return_value = None

    with patch("utils.requests.post", return_value=fake_response) as mock_post:
        result, latency_ms = call_inspect_api(image_path, window_path, "http://fake/inspect")

    assert result == {"status": "healthy", "defect_probability": 0.1}
    assert latency_ms >= 0
    mock_post.assert_called_once()


def test_call_inspect_api_raises_on_http_error(tmp_path):
    image_path = tmp_path / "part.png"
    Image.new("RGB", (10, 10)).save(image_path)
    window_path = tmp_path / "vibration_window.npy"
    np.save(window_path, np.zeros(2048, dtype=np.float32))

    fake_response = MagicMock()
    fake_response.raise_for_status.side_effect = Exception("HTTP 500")

    with patch("utils.requests.post", return_value=fake_response):
        with pytest.raises(Exception):
            call_inspect_api(image_path, window_path, "http://fake/inspect")


def test_correct_when_normal_and_predicted_healthy():
    assert is_prediction_correct(False, None, "healthy", None) is True


def test_incorrect_when_normal_but_predicted_defective():
    assert is_prediction_correct(False, None, "defective", "outer_race") is False


def test_correct_when_defective_and_fault_type_matches():
    assert is_prediction_correct(True, "outer_race", "defective", "outer_race") is True


def test_incorrect_when_defective_but_predicted_healthy():
    assert is_prediction_correct(True, "outer_race", "healthy", None) is False


def test_incorrect_when_defective_but_fault_type_mismatches():
    assert is_prediction_correct(True, "outer_race", "defective", "ball") is False