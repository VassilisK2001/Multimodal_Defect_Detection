from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
 
from defect_detection.data.manifest import load_config
from defect_detection.utils import find_project_root


def _get_n_windows(mat_path: Path, window_size: int) -> int:
    """Number of complete, non-overlapping windows available in a .mat file's DE signal."""
    mat = loadmat(mat_path)
    de_key = [k for k in mat.keys() if "DE_time" in k][0]
    n_samples = mat[de_key].flatten().shape[0]
    return n_samples // window_size
 
 
def _compute_split_blocks(n_windows: int, train_frac: float, val_frac: float) -> dict:
    """Divide a file's window-index range [0, n_windows) into three contiguous,
    non-overlapping blocks: train gets the first train_frac, val gets the next val_frac,
    test gets the remainder."""
    train_end = int(n_windows * train_frac)
    val_end = train_end + int(n_windows * val_frac)
    return {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, n_windows),
    }
 
 
def _stratified_image_split(manifest_df: pd.DataFrame, val_frac: float, test_frac: float,
                             seed: int) -> pd.DataFrame:
    """Assign each manifest row (each row = one MVTec image, already paired with a
    vibration file) to train/val/test, stratified on a combined is_defect + fault_class
    key so all three splits get a representative mix of normal samples and all three
    fault types. Returns the manifest with a new 'split' column added."""
    df = manifest_df.copy()
    df["strat_key"] = df["is_defect"].astype(str) + "_" + df["fault_class"]
 
    train_val_df, test_df = train_test_split(
        df, test_size=test_frac, stratify=df["strat_key"], random_state=seed,
    )
    # val_frac is expressed relative to the full dataset, so rescale it relative to
    # whatever remains after removing the test split
    relative_val_frac = val_frac / (1 - test_frac)
    train_df, val_df = train_test_split(
        train_val_df, test_size=relative_val_frac, stratify=train_val_df["strat_key"],
        random_state=seed,
    )
 
    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")
 
    return pd.concat([train_df, val_df, test_df], axis=0).drop(columns=["strat_key"])
 
 
def _redraw_window_indices(df: pd.DataFrame, project_root: Path, window_size: int,
                            rng: np.random.Generator, train_frac: float,
                            val_frac: float) -> pd.DataFrame:
    """For every row, redraw vibration_window_idx constrained to the block of its
    vibration_file matching its assigned split (train/val/test), so no window index
    can appear in more than one split for the same file. Rows are grouped by
    (vibration_file, split) so all rows sharing a file+split combination draw from the
    same block, with replacement only if the block is smaller than the number of rows
    that need a window from it.
    """
    df = df.copy()
    df["vibration_window_idx"] = -1  
 
    # Cache n_windows per file so it's only computed once, not once per row
    n_windows_cache: dict[str, int] = {}
 
    for (vib_file, split_name), group in df.groupby(["vibration_file", "split"]):
        if vib_file not in n_windows_cache:
            mat_path = project_root / vib_file
            n_windows_cache[vib_file] = _get_n_windows(mat_path, window_size)
        n_windows = n_windows_cache[vib_file]
 
        blocks = _compute_split_blocks(n_windows, train_frac=train_frac, val_frac=val_frac)
        block_start, block_end = blocks[split_name]
        block_size = block_end - block_start
 
        n_needed = len(group)
        replace = n_needed > block_size
 
        drawn_indices = rng.choice(
            np.arange(block_start, block_end), size=n_needed, replace=replace,
        )
        df.loc[group.index, "vibration_window_idx"] = drawn_indices
 
    return df
 
 
def split_manifest(manifest_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Full split pipeline: stratified image-level split, then split-aware window
    index redraw. Returns the manifest with 'split' and corrected 'vibration_window_idx' 
    columns."""
    config = load_config()
    project_root = find_project_root()
    window_size = config["window_size"]
    rng = np.random.default_rng(seed)
 
    train_frac = config["split"]["train_frac"]
    val_frac = config["split"]["val_frac"]
    test_frac = config["split"]["test_frac"]
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "split fractions must sum to 1.0"
 
    split_df = _stratified_image_split(manifest_df, val_frac=val_frac, test_frac=test_frac, seed=seed)
    split_df = _redraw_window_indices(split_df, project_root, window_size, rng,
                                       train_frac=train_frac, val_frac=val_frac)
 
    return split_df.reset_index(drop=True)
 
 
if __name__ == "__main__":
    config = load_config()
    project_root = find_project_root()
 
    manifest_path = project_root / config["paths"]["manifest_dir"] / "manifest.csv"
    manifest_df = pd.read_csv(manifest_path)
 
    result_df = split_manifest(manifest_df)
 
    out_dir = project_root / config["paths"]["manifest_dir"]
    for split_name in ["train", "val", "test"]:
        subset = result_df[result_df.split == split_name].drop(columns=["split"])
        out_path = out_dir / f"{split_name}.csv"
        subset.to_csv(out_path, index=False)
        print(f"{split_name}: {len(subset)} rows -> {out_path}")
        print(subset["is_defect"].value_counts().to_dict())
        print(subset["fault_class"].value_counts().to_dict())
        print()