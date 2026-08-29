from unittest.mock import patch

from defect_detection.data.ingestion import download_dataset, verify_structure

def test_skips_download_when_target_dir_already_has_content(tmp_path):
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    (target_dir / "existing_file.txt").write_text("already here")

    with patch("defect_detection.data.ingestion.kagglehub.dataset_download") as mock_download:
        download_dataset("some/slug", target_dir, "TestDataset")

    mock_download.assert_not_called()


def test_downloads_and_copies_files_and_directories(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "top_level_file.txt").write_text("file content")
    nested_dir = cache_dir / "nested_dir"
    nested_dir.mkdir()
    (nested_dir / "inner_file.txt").write_text("inner content")

    target_dir = tmp_path / "target"

    with patch("defect_detection.data.ingestion.kagglehub.dataset_download", return_value=str(cache_dir)):
        download_dataset("some/slug", target_dir, "TestDataset")

    assert (target_dir / "top_level_file.txt").read_text() == "file content"
    assert (target_dir / "nested_dir" / "inner_file.txt").read_text() == "inner content"


def test_creates_target_dir_if_missing(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "file.txt").write_text("content")

    target_dir = tmp_path / "does_not_exist_yet" / "target"
    assert not target_dir.exists()

    with patch("defect_detection.data.ingestion.kagglehub.dataset_download", return_value=str(cache_dir)):
        download_dataset("some/slug", target_dir, "TestDataset")

    assert target_dir.exists()
    assert (target_dir / "file.txt").exists()


def test_lists_mvtec_categories_excluding_files(tmp_path):
    mvtec_dir = tmp_path / "mvtec"
    mvtec_dir.mkdir()
    (mvtec_dir / "bottle").mkdir()
    (mvtec_dir / "zipper").mkdir()
    (mvtec_dir / "readme.txt").write_text("not a category")

    cwru_dir = tmp_path / "cwru"
    cwru_dir.mkdir()

    result = verify_structure(mvtec_dir, cwru_dir)

    assert set(result["mvtec_categories"]) == {"bottle", "zipper"}


def test_counts_nested_cwru_mat_files(tmp_path):
    mvtec_dir = tmp_path / "mvtec"
    mvtec_dir.mkdir()

    cwru_dir = tmp_path / "cwru"
    nested = cwru_dir / "12k_DE" / "OuterRace"
    nested.mkdir(parents=True)
    (nested / "file1.mat").write_text("x")
    (cwru_dir / "file2.mat").write_text("x")
    (cwru_dir / "readme.txt").write_text("not a mat file")

    result = verify_structure(mvtec_dir, cwru_dir)

    assert result["cwru_mat_file_count"] == 2


def test_handles_missing_directories_without_crashing(tmp_path):
    mvtec_dir = tmp_path / "does_not_exist_mvtec"
    cwru_dir = tmp_path / "does_not_exist_cwru"

    result = verify_structure(mvtec_dir, cwru_dir)

    assert result["mvtec_categories"] == []
    assert result["cwru_mat_file_count"] == 0


def test_returns_correct_dict_keys(tmp_path):
    mvtec_dir = tmp_path / "mvtec"
    mvtec_dir.mkdir()
    cwru_dir = tmp_path / "cwru"
    cwru_dir.mkdir()

    result = verify_structure(mvtec_dir, cwru_dir)

    assert set(result.keys()) == {"mvtec_categories", "cwru_mat_file_count"}