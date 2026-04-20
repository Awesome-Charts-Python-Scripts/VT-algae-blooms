"""
Author: Josh Fishbein
Creates a random forest ensemble model and recommends most important features for a model.

Recommended most important features are:
[
    'vct_day_of_year',
    'vct_water_surface',
    'usgs_conductivity_mean',
    'noaa_precipitation',
    'noaa_wind_speed_mean',
    'noaa_wind_direction_5_min',
    'noaa_air_temp_mean',
    'dec_total nitrogen',
    'dec_total phosphorus',
    'dec_dissolved phosphorus',
    'dec_chlorophyll-a',
    'dec_secchi depth',
    'dec_temperature'
]

Usage:
    python scripts/models/random_forest.py
"""

import numpy as np
import sklearn
import argparse
from pprint import pprint
from datetime import date
from typing import Tuple, List, Dict

from utilities.preprocessing_helpers import (
    get_joined_features_and_targets,
    get_train_test_split,
)
from utilities.evaluation_helpers import (
    generate_precision_recall_curve_plot,
    generate_roc_curve_plot,
)

TARGET = "vct_target_bloom"
TEST_SPLIT_DATE = date(2021, 1, 1)
DEFAULT_BINARY_CLASSIFIER_SCORING = ["roc_auc", "accuracy", "precision", "recall", "f1"]
DEFAULT_MULTICLASS_SCORING = [
    "roc_auc_ovr",
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
]

# Comment out features that are highly correlated with other features
DEFAULT_FEATURE_LIST = [
    "vct_day_of_year",
    #  "vct_latitude",
    #  "vct_longitude",
    #  "vct_water_temp",
    "vct_water_surface",  # We should probably convert this to ordinal and do a trailing mean
    "vct_anabaena",
    "vct_aphanizomenon",
    "vct_microcystin",
    "vct_oscillatoria",
    #  "usgs_water_temp_max",
    #  "usgs_water_temp_min",
    #   "usgs_water_temp_mean",
    #  "usgs_conductivity_max",
    #  "usgs_conductivity_min",
    "usgs_conductivity_mean",
    "noaa_precipitation",
    # 'noaa_air_temp_max',
    # 'noaa_air_temp_min',
    "noaa_wind_speed_mean",
    # 'noaa_wind_speed_2_min',
    "noaa_snow_depth",
    # 'noaa_wind_direction_2_min',
    "noaa_snowfall",
    # 'noaa_wind_speed_5_min',
    "noaa_wind_direction_5_min",
    "noaa_air_temp_mean",
    "dec_total nitrogen",
    "dec_total phosphorus",
    "dec_dissolved phosphorus",
    "dec_chlorophyll-a",
    "dec_secchi depth",
    "dec_temperature",
]


def create_model(predict_test_set: bool = False):
    """Create random forest model

    Args:
        predict_test_set: if provided, runs predictions against the test set. Otherwise, just uses the validation set
    """
    X, y = _get_features_and_targets(TARGET, DEFAULT_FEATURE_LIST)
    X_train, X_test, y_train, y_test = get_train_test_split(X, y)
    if predict_test_set:
        _predict_test_set(X_train, X_test, y_train, y_test)
        return

    groups = [dt.year for dt in y_train.index.get_level_values(0)]
    metrics, predictions = _run_cross_validation(X_train, y_train, groups)
    print(f"--Model performance without feature selection--")
    pprint(metrics, indent=4, sort_dicts=False)

    important_features = _get_important_features(X_train, y_train, groups)
    X, y = _get_features_and_targets(TARGET, important_features)
    X_train, X_test, y_train, y_test = _get_train_test_split(X, y)
    metrics, predictions = _run_cross_validation(X_train, y_train, groups)
    print(f"--Model performance with feature selection--")
    pprint(metrics, indent=4, sort_dicts=False)

    if TARGET == "vct_target_bloom":
        generate_roc_curve_plot(
            "Random Forest",
            y_train,
            predictions[:, 1],
            "figures/random_forest_roc_curve.png",
        )
        generate_precision_recall_curve_plot(
            "Random Forest",
            y_train,
            predictions[:, 1],
            "figures/random_forest_pr_curve.png",
        )

    print(f"--Recommended features to use--\n{important_features}")


def _get_features_and_targets(target: str, feature_list: List[str]):
    df = get_joined_features_and_targets()
    df["dec_temperature"] = df["dec_temperature"].fillna(df["usgs_water_temp_mean"])
    index_cols = ["vct_report_date", "vct_region"]
    df = (
        df[index_cols + feature_list + [target]]
        .dropna()
        .set_index(index_cols)
        .sort_index()
    )
    X = df.drop(columns=target)
    target_dtype = str if df[target].nunique() > 2 else float
    y = df[target].astype(target_dtype)
    return X, y


def _get_train_test_split(X, y):
    X_train = X[X.index.get_level_values(0) < TEST_SPLIT_DATE]
    X_test = X[X.index.get_level_values(0) >= TEST_SPLIT_DATE]
    y_train = y[y.index.get_level_values(0) < TEST_SPLIT_DATE]
    y_test = y[y.index.get_level_values(0) >= TEST_SPLIT_DATE]
    return (X_train, X_test, y_train, y_test)


def _run_cross_validation(
    X, y, groups, scoring=None, n_estimators=1000
) -> Tuple[Dict, np.array]:
    """Run leave one group out cross validation, and return a tuple of the scores and cross validation predictions"""
    if not scoring:
        scoring = (
            DEFAULT_MULTICLASS_SCORING
            if y.nunique() > 2
            else DEFAULT_BINARY_CLASSIFIER_SCORING
        )
    logo = sklearn.model_selection.LeaveOneGroupOut()
    rfc = sklearn.ensemble.RandomForestClassifier(
        n_estimators=n_estimators, max_features="sqrt", random_state=42
    )
    results = sklearn.model_selection.cross_validate(
        rfc, X, y, cv=logo, groups=groups, scoring=scoring, return_estimator=True
    )
    estimators = results["estimator"]
    cv_metrics = {
        k.replace("test_", ""): v.round(3).tolist()
        for k, v in results.items()
        if "test_" in k
    }
    mean_metrics = {
        k.replace("test_", "mean_"): v.mean().round(3)
        for k, v in results.items()
        if "test_" in k
    }
    metrics = {**cv_metrics, **mean_metrics}
    group_indices = [
        [index for index, value in enumerate(groups) if value == grp]
        for grp in sorted(set(groups))
    ]
    cv_predictions = np.concatenate(
        [
            rfc.predict_proba(X.iloc[grp])
            for (rfc, grp) in zip(estimators, group_indices)
        ]
    )
    return metrics, cv_predictions


def _predict_test_set(
    X_train, X_test, y_train, y_test, n_estimators=1000
) -> Tuple[Dict, np.array]:
    rfc = sklearn.ensemble.RandomForestClassifier(
        n_estimators=n_estimators, max_features="sqrt", random_state=42
    )
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    metrics = {
        "roc_auc": sklearn.metrics.roc_auc_score(y_test.values, y_pred),
        "accuracy": sklearn.metrics.accuracy_score(y_test.values, y_pred),
        "precision": sklearn.metrics.precision_score(y_test.values, y_pred),
        "recall": sklearn.metrics.recall_score(y_test.values, y_pred),
        "f1": sklearn.metrics.f1_score(y_test.values, y_pred),
    }

    pprint(metrics, indent=4, sort_dicts=False)

    y_pred_proba = rfc.predict_proba(X_test)
    generate_roc_curve_plot(
        "Random Forest",
        y_test,
        y_pred_proba[:, 1],
        "figures/random_forest_roc_curve.png",
    )
    generate_precision_recall_curve_plot(
        "Random Forest",
        y_test,
        y_pred_proba[:, 1],
        "figures/random_forest_pr_curve.png",
    )


def _get_important_features(X, y, groups, n_estimators=1000, threshold=0.01):
    logo = sklearn.model_selection.LeaveOneGroupOut()
    important_features = {}
    for i, (train, test) in enumerate(logo.split(X, y, groups=groups)):
        rfc = sklearn.ensemble.RandomForestClassifier(
            n_estimators=n_estimators, max_features="sqrt", random_state=i
        )
        X_train = X.iloc[train]
        y_train = y.iloc[train]
        rfc.fit(X_train, y_train)
        for feature, importance in zip(X_train.columns, rfc.feature_importances_):
            if importance < threshold:
                continue
            important_features[feature] = important_features.get(feature, []) + [
                importance
            ]
    return [
        feat
        for feat, importances in important_features.items()
        if len(importances) == len(set(groups))
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Generate a random forest model and print performance metrics"
    )
    parser.add_argument(
        "--predict-test",
        dest="predict_test",
        action="store_true",
        default=False,
        help="Predict the test set",
    )
    args = parser.parse_args()
    create_model(predict_test_set=args.predict_test)


if __name__ == "__main__":
    main()
