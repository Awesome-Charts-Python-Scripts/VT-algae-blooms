"""
Author: Charlie Davidson
Purpose: Use Logistic Regression to predict binary algae bloom target
"""

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneGroupOut, cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

from utilities.preprocessing_helpers import get_joined_features_and_targets

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

# Fill in missing numeric values with median
# Scale data
numeric_transformer = Pipeline(
    [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

# Pipeline
pipeline = Pipeline(
    [("preprocess", numeric_transformer), ("model", LogisticRegression(max_iter=1000))]
)

# Cross validation (Leave one year out per fold)
years = train["vct_report_date"].dt.year

X_train = train[feature_cols]
y_train = train["vct_target_bloom"]

leave_one_out = LeaveOneGroupOut()

cv_results = cross_validate(
    pipeline,
    X_train,
    y_train,
    groups=years,
    cv=leave_one_out,
    scoring=["roc_auc", "accuracy", "precision", "recall", "f1"],
    return_train_score=False,
)

# Scores for each cross-validation fold:
for metric, values in cv_results.items():
    if metric.startswith("test_"):
        print(metric, values.mean().round(4), values.round(4))

# test_roc_auc 0.6806 [0.8306 0.698  0.663  0.7317 0.5866 0.5737]
# test_accuracy 0.7793 [0.8006 0.8117 0.7939 0.8012 0.7012 0.767 ]
# test_precision 0.5198 [0.5357 0.5082 0.575  0.7917 0.3333 0.375 ]
# test_recall 0.2525 [0.4    0.5254 0.2875 0.2405 0.0242 0.0375]
# test_f1 0.3067 [0.458  0.5167 0.3833 0.3689 0.0451 0.0682]

# Plot ROC Curve
y_out_of_fold_prob = cross_val_predict(
    pipeline, X_train, y_train, groups=years, cv=leave_one_out, method="predict_proba"
)[:, 1]

cv_auc = roc_auc_score(y_train, y_out_of_fold_prob)
fpr, tpr, _ = roc_curve(y_train, y_out_of_fold_prob)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"ROC (AUC = {cv_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Logistic Regression ROC Curve (Out-of-Fold)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("figures/logistic_regression_roc_curve.png", dpi=300, bbox_inches="tight")
plt.show()

# Plot Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_train, y_out_of_fold_prob)

ap = average_precision_score(y_train, y_out_of_fold_prob)

plt.figure(figsize=(6, 6))
plt.plot(recall, precision, label=f"AP = {ap:.3f}")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Logistic Regression Precision–Recall Curve")
plt.legend()
plt.grid(True)

plt.savefig("figures/logistic_regression_pr_curve.png", dpi=300, bbox_inches="tight")
plt.show()
