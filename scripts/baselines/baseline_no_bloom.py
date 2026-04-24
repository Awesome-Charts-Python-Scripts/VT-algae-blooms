"""
Author: Josh Fishbein
Creates a baseline prediction of always false for comparison of accuracy metrics

Usage:
    python scripts/baselines/baseline_no_bloom.py
"""

import sklearn
import numpy as np
from pprint import pprint

from utilities.preprocessing_helpers import (
    get_joined_features_and_targets,
    get_train_test_split,
)
from utilities.evaluation_helpers import (
    generate_precision_recall_curve_plot,
    generate_roc_curve_plot,
)

TARGET = "vct_target_bloom"


def create_model():
    """Baseline model that always predicts "no bloom"""
    df = get_joined_features_and_targets().set_index(["vct_region", "vct_report_date"])
    X = df.drop(columns=TARGET)
    y = df[[TARGET]]
    _, _, _, y_test = get_train_test_split(X, y)
    y_pred = np.array([0.0] * len(y_test))

    metrics = {
        "roc_auc": sklearn.metrics.roc_auc_score(y_test.values, y_pred),
        "accuracy": sklearn.metrics.accuracy_score(y_test.values, y_pred),
        "precision": sklearn.metrics.precision_score(y_test.values, y_pred),
        "recall": sklearn.metrics.recall_score(y_test.values, y_pred),
        "f1": sklearn.metrics.f1_score(y_test.values, y_pred),
    }

    pprint(metrics, indent=4, sort_dicts=False)

    generate_roc_curve_plot(
        "Baseline (no bloom)",
        y_test[TARGET].astype(float),
        y_pred.astype(float),
        "figures/baseline_no_bloom_roc_curve.png",
    )
    generate_precision_recall_curve_plot(
        "Baseline (no bloom)",
        y_test[TARGET].astype(float),
        y_pred.astype(float),
        "figures/baseline_no_bloom_pr_curve.png",
    )


if __name__ == "__main__":
    create_model()
