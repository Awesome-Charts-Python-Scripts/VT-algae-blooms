"""
Support Vector Technique implementations

Alex Schaefer
"""

import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
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
    (df["vct_report_date"] >= "2015-01-01") &
    (df["vct_report_date"] <  "2021-01-01")
]

test = df[
    (df["vct_report_date"] >= "2021-01-01") &
    (df["vct_report_date"] <= "2022-12-31")
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
    "dec_temperature"
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

# Begin loop
model_types = ['linear', 'poly', 'rbf', 'sigmoid']
for model_type in model_types:

    # Pipeline
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")), # Fill in missing with median
        ("scaler", StandardScaler()), # standard scaling
        ("model", SVC(kernel=model_type, gamma='auto'))
    ])

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
        return_train_score=False
    )

    # print(cv_results)

    print(f"Results for SVM with model type {model_type}---------")

    # Scores for each cross-validation fold:
    for metric, values in cv_results.items():
        if metric.startswith("test_"):
            print(metric, values.mean().round(4), values.round(4))

