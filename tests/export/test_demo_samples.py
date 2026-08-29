import numpy as np
import pandas as pd
from PIL import Image
from scipy.io import savemat

from tests.factories import CLASS_NAMES
from defect_detection.export.demo_samples import build_manifest_entry, select_demo_rows, write_demo_sample


def _make_test_df():
    rows = (
        [{"is_defect": 0, "fault_class": None}] * 5
        + [{"is_defect": 1, "fault_class": "outer_race"}] * 4
        + [{"is_defect": 1, "fault_class": "inner_race"}] * 2
        + [{"is_defect": 1, "fault_class": "ball"}] * 1
    )
    return pd.DataFrame(rows)


def test_selects_correct_scenarios_and_counts():
    test_df = _make_test_df()

    result = select_demo_rows(test_df, CLASS_NAMES, n_per_class=2)

    assert set(result.keys()) == {"normal", "outer_race", "inner_race", "ball"}
    assert len(result["normal"]) == 2
    assert len(result["outer_race"]) == 2
    assert len(result["inner_race"]) == 2


def test_returns_fewer_than_n_per_class_when_insufficient_candidates():
    test_df = _make_test_df()

    result = select_demo_rows(test_df, CLASS_NAMES, n_per_class=2)

    # ball only has 1 candidate row, less than n_per_class=2.
    assert len(result["ball"]) == 1


def test_prefers_preferred_indices_over_random():
    test_df = _make_test_df()
    outer_race_indices = test_df[test_df.fault_class == "outer_race"].index.tolist()
    preferred = [outer_race_indices[0]]

    result = select_demo_rows(test_df, CLASS_NAMES, n_per_class=1, preferred_row_indices=preferred)

    assert result["outer_race"].index.tolist() == preferred


def test_tops_up_with_random_when_fewer_preferred_than_n():
    test_df = _make_test_df()
    outer_race_indices = test_df[test_df.fault_class == "outer_race"].index.tolist()
    preferred = [outer_race_indices[0]]

    result = select_demo_rows(test_df, CLASS_NAMES, n_per_class=3, preferred_row_indices=preferred)

    assert len(result["outer_race"]) == 3
    assert preferred[0] in result["outer_race"].index.tolist()


def test_deterministic_given_same_seed():
    test_df = _make_test_df()

    result_a = select_demo_rows(test_df, CLASS_NAMES, n_per_class=2, seed=7)
    result_b = select_demo_rows(test_df, CLASS_NAMES, n_per_class=2, seed=7)

    assert result_a["normal"].index.tolist() == result_b["normal"].index.tolist()


def test_write_demo_sample_writes_correct_image_and_window(tmp_path):
    source_image_path = tmp_path / "source.png"
    Image.new("RGB", (32, 32), color=(10, 20, 30)).save(source_image_path)

    signal = np.arange(100, dtype=np.float64)
    savemat(tmp_path / "signal.mat", {"X097_DE_time": signal.reshape(-1, 1)})

    row = pd.Series({
        "image_path": "source.png", "vibration_file": "signal.mat", "vibration_window_idx": 1,
    })
    output_dir = tmp_path / "demo_out"

    write_demo_sample(row, tmp_path, window_size=10, output_dir=output_dir)

    assert (output_dir / "part.png").exists()
    written_image = Image.open(output_dir / "part.png")
    assert written_image.size == (32, 32)

    written_window = np.load(output_dir / "vibration_window.npy")
    assert np.array_equal(written_window, signal[10:20].astype(np.float32))


def test_manifest_entry_for_normal_row_has_no_fault_class():
    row = pd.Series({"is_defect": 0, "fault_class": None})

    entry = build_manifest_entry("normal", "sample_1", row)

    assert entry["true_is_defect"] is False
    assert entry["true_fault_class"] is None


def test_manifest_entry_for_defective_row_includes_fault_class():
    row = pd.Series({"is_defect": 1, "fault_class": "outer_race"})

    entry = build_manifest_entry("outer_race", "sample_1", row)

    assert entry["true_is_defect"] is True
    assert entry["true_fault_class"] == "outer_race"