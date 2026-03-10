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

merged_data = "data/merged_csvs/algae_merged.csv"

# Load CSV
df = pd.read_csv(merged_data)

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
    "vct_water_temp_7days",
    "vct_water_surface_7days",
    "vct_anabaena_7days",
    "vct_aphanizomenon_7days",
    "vct_microcystin_7days",
    "vct_oscillatoria_7days",
    "vct_water_temp_14days",
    "vct_water_surface_14days",
    "vct_anabaena_14days",
    "vct_aphanizomenon_14days",
    "vct_microcystin_14days",
    "vct_oscillatoria_14days",
    "usgs_water_temp_max_trailing",
    "usgs_water_temp_min_trailing",
    "usgs_water_temp_mean_trailing",
    "usgs_conductivity_max_trailing",
    "usgs_conductivity_min_trailing",
    "usgs_conductivity_mean_trailing",
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

# categorical_cols = ["vct_water_surface_trailing"]

# numeric_cols = [
#     c for c in feature_cols
#     if c not in categorical_cols
# ]

# Fill in missing numeric values with median
# Scale data
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Fill in missing categorical values with most frequent value
# One Hot Encoding
# categorical_transformer = Pipeline([
#     ("imputer", SimpleImputer(strategy="most_frequent")),
#     ("onehot", OneHotEncoder(handle_unknown="ignore"))
# ])

#Preprocessor
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, feature_cols),
    # ("cat", categorical_transformer, categorical_cols)
])

# Pipeline
pipeline = Pipeline([
    ("preprocess", preprocessor),
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

print("CV AUC:", np.mean(scores))
# CV AUC: 0.6719222679761537