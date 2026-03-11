"""
Author: Charlie Davidson
Purpose: Use Logistic Regression to predict binary algae bloom target
"""

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

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

# Fill in missing numeric values with median
# Scale data
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Pipeline
pipeline = Pipeline([
    ("preprocess", numeric_transformer),
    ("model", LogisticRegression(max_iter=1000))
])

# Cross-validation
years = sorted(train["vct_report_date"].dt.year.unique())

scores = []

for year in years:

    train_fold = train[train["vct_report_date"].dt.year != year]
    val_fold   = train[train["vct_report_date"].dt.year == year]

    X_train = train_fold[feature_cols]
    y_train = train_fold["vct_target_bloom"]

    X_val = val_fold[feature_cols]
    y_val = val_fold["vct_target_bloom"]

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict_proba(X_val)[:,1]

    score = roc_auc_score(y_val, preds)

    scores.append(score)

print(np.round(scores,4))
print("CV AUC:", np.round(np.mean(scores), 4))
# [0.8072 0.7237 0.6207 0.6931 0.5943 0.5605]
# CV AUC: 0.6666