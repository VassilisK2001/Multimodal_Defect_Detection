import pandas as pd
from PIL import Image

from defect_detection.data.image_io import load_images_for_df


def test_load_images_for_df_returns_correct_count_and_shape(tmp_path):
    for i in range(3):
        Image.new("RGB", (64, 64), color=(i * 10, 0, 0)).save(tmp_path / f"img_{i}.png")

    df = pd.DataFrame({"image_path": [f"img_{i}.png" for i in range(3)]})

    result = load_images_for_df(df, tmp_path)

    assert result.shape[0] == 3
    assert result.shape[1] == 3  # channels