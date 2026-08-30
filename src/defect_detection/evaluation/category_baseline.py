"""
Category-only baseline: trains a simple classifier predicting fault_class
from MVTec category alone (no image or vibration input), to test whether
the image branch's fault-type contribution could be explained by category
recognition rather than genuine defect appearance.
"""


from typing import cast

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

from defect_detection.evaluation.metrics import compute_fault_type_metrics


def predict_category_only_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame,
                                    class_names: list[str], category_column: str = "category",
                                    seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Fit a category-only classifier and predict on the test set.

    Args:
        train_df: Training split manifest.
        test_df: Test split manifest.
        class_names: Fault class names, in index order.
        category_column: Manifest column holding the MVTec category.
        seed: Random seed for the classifier.

    Returns:
        (y_test, y_pred): true and predicted fault_class indices, in the order of
        test_df's defective rows.
    """
    train_defective = train_df[train_df.is_defect == 1]
    test_defective = test_df[test_df.is_defect == 1]

    encoder = OneHotEncoder(handle_unknown="ignore")
    X_train = encoder.fit_transform(train_defective[[category_column]])
    X_test = encoder.transform(test_defective[[category_column]])

    fault_class_to_idx = {name: i for i, name in enumerate(class_names)}
    train_fault_class = cast(pd.Series, train_defective["fault_class"])
    test_fault_class = cast(pd.Series, test_defective["fault_class"])
    y_train = cast(np.ndarray, train_fault_class.map(fault_class_to_idx).values)
    y_test = cast(np.ndarray, test_fault_class.map(fault_class_to_idx).values)

    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(X_train, y_train)
    y_pred = cast(np.ndarray, clf.predict(X_test))

    return y_test, y_pred


def train_category_only_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame,
                                  class_names: list[str], category_column: str = "category",
                                  seed: int = 42) -> dict:
    """Train a category-only baseline and compute its fault-type metrics.

    Args:
        train_df: Training split manifest.
        test_df: Test split manifest.
        class_names: Fault class names, in index order.
        category_column: Manifest column holding the MVTec category.
        seed: Random seed for the classifier.

    Returns:
        The output of compute_fault_type_metrics()
    """
    y_test, y_pred = predict_category_only_baseline(train_df, test_df, class_names, category_column, seed)
    return compute_fault_type_metrics(y_test, y_pred, class_names)


def print_category_comparison(category_baseline: dict, image_only: dict,
                               both_zero: dict, both_shuffle: dict) -> None:
    """Print per-class F1 and macro-F1 side by side across four fault-type sources.

    Args:
        category_baseline: Output of train_category_only_baseline().
        image_only: The "fault_metrics" dict from the image-only model's saved
            evaluation results.
        both_zero: The "fault_metrics" dict from the fusion model's
            vibration_corrupted (zero method) modality shuffle results.
        both_shuffle: The "fault_metrics" dict from the fusion model's
            vibration_corrupted (shuffle method) modality shuffle results.
    """
    sources = {
        "category-only": category_baseline,
        "image-only": image_only,
        "both (zero)": both_zero,
        "both (shuffle)": both_shuffle,
    }
    class_names = list(category_baseline["per_class"].keys())

    print(f"{'':16}" + "".join(f"{name:>16}" for name in sources))
    for class_name in class_names:
        row_values = "".join(
            f"{sources[name]['per_class'][class_name]['f1']:>16.3f}" for name in sources
        )
        print(f"{class_name:16}{row_values}")

    macro_values = "".join(f"{sources[name]['macro_f1']:>16.3f}" for name in sources)
    print(f"{'macro_f1':16}{macro_values}")


def compare_predictions_by_category(test_df: pd.DataFrame, y_true: np.ndarray,
                                     both_pred: np.ndarray, category_pred: np.ndarray,
                                     category_column: str = "category") -> pd.DataFrame:
    """Break down prediction accuracy by MVTec category, and how often the fusion
    model repeats the category-only baseline's mistakes specifically.

    Args:
        test_df: Test split manifest (defective rows only, same row order as
            y_true/both_pred/category_pred).
        y_true: True fault_class indices, per defective test row.
        both_pred: The fusion model's vibration-corrupted fault_class predictions.
        category_pred: The category-only baseline's fault_class predictions.
        category_column: Manifest column holding the MVTec category.

    Returns:
        DataFrame indexed by category: n_samples, both_accuracy, category_accuracy,
        and agreement_on_category_errors.
    """
    test_defective = test_df[test_df.is_defect == 1].reset_index(drop=True)
    df = pd.DataFrame({
        "category": test_defective[category_column].values,
        "y_true": y_true,
        "both_pred": both_pred,
        "category_pred": category_pred,
    })

    rows = []
    for category, group in df.groupby("category"):
        category_wrong = group[group["category_pred"] != group["y_true"]]
        agreement_on_errors = (
            (category_wrong["both_pred"] == category_wrong["category_pred"]).mean()
            if len(category_wrong) > 0 else float("nan")
        )
        rows.append({
            "category": category,
            "n_samples": len(group),
            "both_accuracy": (group["both_pred"] == group["y_true"]).mean(),
            "category_accuracy": (group["category_pred"] == group["y_true"]).mean(),
            "agreement_on_category_errors": agreement_on_errors,
        })

    return pd.DataFrame(rows).set_index("category")