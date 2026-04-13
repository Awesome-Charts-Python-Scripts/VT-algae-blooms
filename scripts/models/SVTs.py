"""
Support Vector Technique implementations

Alex Schaefer
"""

import pandas as pd
import numpy as np
from typing import List

import sklearn
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

from sklearn.metrics import PrecisionRecallDisplay
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneGroupOut, cross_validate
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt

from utilities.preprocessing_helpers import get_joined_features_and_targets

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    average_precision_score,
    f1_score,
)
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, roc_curve
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
)


def _generate_roc_curve_plot(y, y_pred, model):
    cv_auc = sklearn.metrics.roc_auc_score(y, y_pred)
    false_positive_rate, true_positive_rate, _ = sklearn.metrics.roc_curve(y, y_pred)
    plt.figure(figsize=(6, 6))
    plt.plot(false_positive_rate, true_positive_rate, label=f"ROC (AUC = {cv_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model} ROC Curve (Out-of-Fold)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"figures/{model}_roc_curve.png", dpi=300, bbox_inches="tight")


def _generate_precision_recall_curve_plot(y, y_pred, model):
    precision, recall, _ = sklearn.metrics.precision_recall_curve(y, y_pred)
    average_precision = sklearn.metrics.average_precision_score(y, y_pred)
    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label=f"Average Precision = {average_precision:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{model} Precision–Recall Curve")
    plt.legend()
    plt.grid(True)
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    x_pad = x_max - 1
    y_pad = y_max - 1
    plt.xlim(-x_pad, 1 + x_pad)
    plt.ylim(-y_pad, 1 + y_pad)
    plt.savefig(f"figures/{model}_pr_curve.png", dpi=300, bbox_inches="tight")


merged_data = "data/merged_csvs/algae_merged.csv"

# Load CSV
df = get_joined_features_and_targets()


# Train/test split
df["vct_report_date"] = pd.to_datetime(df["vct_report_date"])

train = df[
    (df["vct_report_date"] >= "2015-01-01") & (df["vct_report_date"] < "2021-01-01")
]

test = df[
    (df["vct_report_date"] >= "2021-01-01") & (df["vct_report_date"] <= "2022-12-31")
]

TARGET = "vct_target_bloom"


# Feature columns
feature_cols = [
    "vct_water_temp",
    "vct_water_surface",
    "vct_anabaena",
    "vct_aphanizomenon",
    "vct_microcystin",
    "vct_oscillatoria",
    "usgs_water_temp_max",
    "usgs_water_temp_min",
    "usgs_water_temp_mean",
    "usgs_conductivity_max",
    "usgs_conductivity_min",
    "usgs_conductivity_mean",
    "noaa_precipitation",
    "noaa_air_temp_max",
    "noaa_air_temp_min",
    "noaa_wind_speed_mean",
    "noaa_wind_speed_2_min",
    "noaa_snow_depth",
    "noaa_wind_direction_2_min",
    "noaa_snowfall",
    "noaa_wind_speed_5_min",
    "noaa_wind_direction_5_min",
    "noaa_air_temp_mean",
    "dec_total nitrogen",
    "dec_total phosphorus",
    "dec_dissolved phosphorus",
    "dec_chlorophyll-a",
    "dec_secchi depth",
    "dec_temperature",
]

feature_cols = [
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


from datetime import date

TEST_SPLIT_DATE = date(2021, 1, 1)


def _get_train_test_split(X, y):
    X_train = X[X.index.get_level_values(0) < TEST_SPLIT_DATE]
    X_test = X[X.index.get_level_values(0) >= TEST_SPLIT_DATE]
    y_train = y[y.index.get_level_values(0) < TEST_SPLIT_DATE]
    y_test = y[y.index.get_level_values(0) >= TEST_SPLIT_DATE]
    return (X_train, X_test, y_train, y_test)


# Begin loop
model_types = ["linear", "poly", "rbf", "sigmoid"]
X, y = _get_features_and_targets(TARGET, feature_cols)
X_train, X_test, y_train, y_test = _get_train_test_split(X, y)
groups = [dt.year for dt in y_train.index.get_level_values(0)]
for model_type in model_types:
    svc = SVC(kernel=model_type, gamma="auto")
    # Pipeline
    pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),  # Fill in missing with median
            ("scaler", StandardScaler()),  # standard scaling
            ("model", svc),
        ]
    )

    leave_one_out = LeaveOneGroupOut()

    cv_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        groups=groups,
        cv=leave_one_out,
        scoring=["roc_auc", "accuracy", "precision", "recall", "f1"],
        return_train_score=True,
        return_estimator=True,
    )
    # print(f"fitting {model_type}")
    # si = SimpleImputer(strategy="median")
    # X_train = si.fit_transform(X_train)
    # X_test = si.fit_transform(X_test)

    # sc = StandardScaler()
    # X_train = sc.fit_transform(X_train)
    # X_test = sc.transform(X_test)
    # svc.fit(X_train, y_train)

    # # print(cv_results)

    # estimators = cv_results["estimator"]

    # print(f"Generating plots for {model_type}:")

    # display = PrecisionRecallDisplay.from_estimator(
    #     svc, X_test, y_test, name=f"{model_type}SVC", despine=True
    # )
    # _ = display.ax_.set_title(f"2-class Precision-Recall curve for {model_type} SVC")

    # plt.savefig(f"figures/SVT_{model_type}_pr_curve.png", dpi=300, bbox_inches="tight")

    # from sklearn.metrics import RocCurveDisplay

    # RocCurveDisplay.from_estimator(
    # svc, X_test, y_test, plot_chance_level=True)
    # plt.savefig(f"figures/SVT_{model_type}_ROC_curve.png", dpi=300, bbox_inches="tight")

    # print(f"Results for SVM with model type {model_type}---------")

    # # Scores for each cross-validation fold:
    # for metric, values in cv_results.items():
    #     if metric.startswith("test_"):
    #         print(metric, values.mean().round(4), values.round(4))

    # _generate_precision_recall_curve_plot(y_train, cv_results, model_type)
    # _generate_roc_curve_plot(y_train, cv_results, model_type)
