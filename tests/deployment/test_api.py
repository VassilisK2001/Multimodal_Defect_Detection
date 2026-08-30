"""
Tests for src/defect_detection/deployment/api.py.
"""


import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

DEPLOYMENT_DIR = Path(__file__).resolve().parents[2] / "deployment"
sys.path.insert(0, str(DEPLOYMENT_DIR))

from api import app

DEMO_SAMPLES_DIR = DEPLOYMENT_DIR / "demo_samples"

def _find_first_sample(scenario: str) -> Path:
    sample_dirs = sorted((DEMO_SAMPLES_DIR / scenario).glob("sample_*"))
    if not sample_dirs:
        pytest.skip(f"No demo samples found for scenario '{scenario}'. Run scripts/build_demo_samples.py first.")
    return sample_dirs[0]


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_inspect_returns_valid_response_schema(client):
    sample_dir = _find_first_sample("normal")

    with open(sample_dir / "part.png", "rb") as image_file, \
         open(sample_dir / "vibration_window.npy", "rb") as window_file:
        response = client.post(
            "/inspect",
            files={
                "image": ("part.png", image_file, "image/png"),
                "vibration_window": ("vibration_window.npy", window_file, "application/octet-stream"),
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] in ("healthy", "defective")
    assert isinstance(body["defect_probability"], float)
    assert 0.0 <= body["defect_probability"] <= 1.0


def test_inspect_defective_sample_includes_fault_type(client):
    sample_dir = _find_first_sample("outer_race")

    with open(sample_dir / "part.png", "rb") as image_file, \
         open(sample_dir / "vibration_window.npy", "rb") as window_file:
        response = client.post(
            "/inspect",
            files={
                "image": ("part.png", image_file, "image/png"),
                "vibration_window": ("vibration_window.npy", window_file, "application/octet-stream"),
            },
        )

    assert response.status_code == 200
    body = response.json()
    if body["status"] == "defective":
        assert body["fault_type"] in ("outer_race", "inner_race", "ball")
        assert 0.0 <= body["fault_confidence"] <= 1.0
    else:
        assert body["fault_type"] is None


def test_inspect_missing_file_returns_422(client):
    sample_dir = _find_first_sample("normal")

    with open(sample_dir / "part.png", "rb") as image_file:
        response = client.post("/inspect", files={"image": ("part.png", image_file, "image/png")})

    assert response.status_code == 422