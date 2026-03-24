"""
Author: Helen Flannery
Project: CS5540 Spring 2026 Machine Learning class project
Purpose: Use Linear Regression to predict:
    (a) algae bloom presence (binary values)
    (b) algae bloom intensity (continuous values from 1-3)
"""

import os
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate, KFold, LeaveOneGroupOut

from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, ElasticNetCV

from sklearn.metrics import accuracy_score, precision_score, recall_score, average_precision_score, f1_score
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, roc_curve
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import confusion_matrix

# ---------------------------------------------------------------
# Set up logging
# ---------------------------------------------------------------

log_directory = r"..\..\logs"
log_path = os.path.join(log_directory, "linear_regression.log")

# Set up logging to print to terminal and to log file
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(log_path, mode="w"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger()

log.info(f"Running linear regression\n")

# ---------------------------------------------------------------
# Load source data
# ---------------------------------------------------------------

# Source data info
data_directory = r"..\..\data\merged_csvs"
algae_file = "algae_merged"
algae_path = os.path.join(data_directory, algae_file + ".csv")

# Read in source data
algae_raw_df = pd.read_csv(algae_path)

# ---------------------------------------------------------------
# Set figure output directory
# ---------------------------------------------------------------

fig_directory = r"..\..\figures"

# ---------------------------------------------------------------
# Convert columns to floats, integers, and dates as appropriate
# ---------------------------------------------------------------

algae_numcols_df = algae_raw_df.copy()

date_cols = ["vct_report_date"]

string_cols = ["vct_region"]

integer_cols = [
    "vct_anabaena_7days",
    "vct_aphanizomenon_7days",
    "vct_microcystin_7days",
    "vct_oscillatoria_7days",
    "vct_anabaena_14days",
    "vct_aphanizomenon_14days",
    "vct_microcystin_14days",
    "vct_oscillatoria_14days",
    "vct_target_bloom",
    "vct_target_bloom_7days",
    "vct_target_bloom_14days",
    "vct_day_of_year",
    "vct_year"
]

# Format date columns
algae_numcols_df[date_cols] = (
    algae_numcols_df[date_cols]
    .apply(pd.to_datetime, errors="coerce")
)

# Format string columns
algae_numcols_df[string_cols] = algae_numcols_df[string_cols].astype("string")

# Format integer columns (nullable)
algae_numcols_df[integer_cols] = (
    algae_numcols_df[integer_cols]
        .apply(pd.to_numeric, errors="coerce")
        .astype("Int64")
)

# Format all remaining columns as floats
float_cols = algae_numcols_df.columns.difference(string_cols + integer_cols + date_cols)

algae_numcols_df[float_cols] = (
    algae_numcols_df[float_cols]
        .apply(pd.to_numeric, errors="coerce")
        .astype(float)
)

# Remove 2014 (only five observations)
algae_goodyears_df = algae_numcols_df[algae_numcols_df["vct_year"] != 2014]

# ---------------------------------------------------------------
# Define column types
# ---------------------------------------------------------------

algae_model_df = algae_goodyears_df.copy()

target_cols = ["vct_target_bloom", "vct_target_bloom_intensity"]

meta_cols = ["vct_report_date", "vct_year"]

exclude_cols = ["vct_region", "vct_latitude", "vct_longitude"]

trailing_7day_cols = [
    "vct_anabaena_7days",
    "vct_aphanizomenon_7days",
    "vct_microcystin_7days",
    "vct_oscillatoria_7days",
    "vct_water_temp_7days",
    "vct_water_surface_7days",
    "usgs_conductivity_mean_7days",
    # "usgs_water_temp_mean_7days"
]

trailing_14day_cols = [
    "vct_anabaena_14days",
    "vct_aphanizomenon_14days",
    "vct_microcystin_14days",
    "vct_oscillatoria_14days",
    "vct_water_temp_14days",
    "vct_water_surface_14days",
    "usgs_conductivity_mean_14days",
    "usgs_water_temp_mean_14days"
]

trailing_target_cols = [
    "vct_target_bloom_7days",
    "vct_target_bloom_14days",
    "vct_target_bloom_intensity_7days",
    "vct_target_bloom_intensity_14days"
]

correlated_cols = ["dec_temperature", 
                   "usgs_water_temp_max_7days", "usgs_water_temp_min_7days",
                   "usgs_water_temp_max_14days", "usgs_water_temp_min_14days",
                   "usgs_conductivity_max_7days", "usgs_conductivity_min_7days",
                   "usgs_conductivity_max_14days", "usgs_conductivity_min_14days",
                   "noaa_air_temp_max", "noaa_air_temp_min", "noaa_air_temp_mean",
                   "dec_dissolved phosphorus",
                   "usgs_water_temp_mean_7days"
]

# Define unhelpful columns based on coefficient testing later on in the notebook (for feature pruning)
unhelpful_cols = ["noaa_snow_depth", "noaa_snowfall", "noaa_precipitation",
                  "noaa_wind_speed_2_min", "noaa_wind_speed_5_min", "noaa_wind_speed_mean", 
                  "noaa_wind_direction_5_min", 
                  # "noaa_wind_direction_2_min",
                  "vct_water_surface_7days", 
                  # "vct_oscillatoria_7days", "vct_aphanizomenon_7days",
                  "usgs_conductivity_mean_7days",
                  # "usgs_water_temp_mean_7days",
                  "dec_secchi depth", 
                  # "dec_total nitrogen"
]

# Include only 7-day trailing cols in features
# And do not include trailing_target_cols in features yet
other_cols = algae_model_df.columns.difference(meta_cols + correlated_cols + trailing_7day_cols + trailing_14day_cols + trailing_target_cols + target_cols + exclude_cols)
full_feature_cols = trailing_7day_cols + list(other_cols)
# feature_cols = trailing_7day_cols + list(other_cols)
feature_cols = [col for col in full_feature_cols if col not in unhelpful_cols]

log.info(f"Feature columns:\n{feature_cols}\n")

# ---------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------

# Define years to hold out for testing
test_years = [2021, 2022]

# Training set should be all rows *not* in the test years
lr_train_set = algae_model_df[~algae_model_df["vct_year"].isin(test_years)].copy()

# Test set should be all rows *in* the test years
lr_test_set = algae_model_df[algae_model_df["vct_year"].isin(test_years)].copy()

# ---------------------------------------------------------------
# Training features for bloom presence and bloom intensity models
# ---------------------------------------------------------------

# Create a dataframe of features 
# and two series of the target data: bloom presence and bloom intensity
# and a dataframe for the meta information aligned with features by index
algae_train_features = lr_train_set[feature_cols].copy()
algae_train_target_bloom = lr_train_set["vct_target_bloom"].copy()
algae_train_target_intensity = lr_train_set["vct_target_bloom_intensity"].copy()
algae_train_meta = lr_train_set[meta_cols].copy()

# Force all feature columns to be numeric
X_train_features_df = algae_train_features.apply(pd.to_numeric, errors="coerce")

# Convert dataframe into a matrix and targets into vectors 

X_lr_train_features = X_train_features_df.to_numpy() # converts feature dataframe into an array
y_lr_train_bloom = np.asarray(algae_train_target_bloom) # converts target series into a vector
y_lr_train_intensity = np.asarray(algae_train_target_intensity) # converts target series into a vector

# Find the number of features
n_features = X_train_features_df.shape[1]

# Add a bias (intercept) term of all 1s for linear regressoin
# Update: don't need if setting linreg parameter fit_intercept = true
# X_lr_features = np.c_[np.ones(X_lr_features.shape[0]), X_lr_features]

# Initialize parameters (theta) into an all-zero matrix
# lr_theta = np.zeros((n_features, 1))
lr_train_theta = np.zeros((X_train_features_df.shape[1], 1))

# Print shapes
log.info("Training data set shapes:")
log.info(f"X_train_features_df shape: {X_train_features_df.shape}")
log.info(f"y_lr_train_bloom shape: {y_lr_train_bloom.shape}")
log.info(f"y_lr_train_intensity shape: {y_lr_train_intensity.shape}")
log.info(f"lr_train_theta shape: {lr_train_theta.shape}\n")

# ---------------------------------------------------------------
# Test features for bloom presence and bloom intensity models
# ---------------------------------------------------------------

# Create a dataframe of features 
# and two series of the target data: bloom presence and bloom intensity
# and a dataframe for the meta information aligned with features by index

algae_test_features = lr_test_set[feature_cols].copy()
algae_test_target_bloom = lr_test_set["vct_target_bloom"].copy()
algae_test_target_intensity = lr_test_set["vct_target_bloom_intensity"].copy()
algae_test_meta = lr_test_set[meta_cols].copy()

# Force all feature columns to be numeric
X_test_features_df = algae_test_features.apply(pd.to_numeric, errors="coerce")

# Convert dataframe into a matrix and targets into vectors 
X_lr_test_features = X_test_features_df.to_numpy() # converts feature dataframe into an array
y_lr_test_bloom = np.asarray(algae_test_target_bloom) # converts target series into a vector
y_lr_test_intensity = np.asarray(algae_test_target_intensity) # converts target series into a vector

# Find the number of features
n_features = X_test_features_df.shape[1]

# Initialize parameters (theta) into an all-zero matrix
# lr_theta = np.zeros((n_features, 1))
lr_test_theta = np.zeros((X_test_features_df.shape[1], 1))

# Print shapes
log.info("Test data set shapes:")
log.info(f"X_test_features_df shape: {X_test_features_df.shape}")
log.info(f"y_lr_test_bloom shape: {y_lr_test_bloom.shape}")
log.info(f"y_lr_test_intensity shape: {y_lr_test_intensity.shape}")
log.info(f"lr_test_theta shape: {lr_test_theta.shape}\n")

# ---------------------------------------------------------------
# Fill NaN columns
# ---------------------------------------------------------------

# Find columns with NaN values for training data
train_cols_with_nans = X_train_features_df.columns[X_train_features_df.isna().any()]
train_nan_cols = []
log.info("Training data set columns with NaNs:")
for col in train_cols_with_nans:
    train_nan_cols.append(col)
    log.info(f"{col}-{X_train_features_df[col].isna().sum()} NaNs")

# Fill in NaN columns in the training data with means
X_train_features_df[train_nan_cols] = X_train_features_df[train_nan_cols].fillna(X_train_features_df[train_nan_cols].mean())

# Save training means for using in imputing test data
train_means = X_train_features_df[train_nan_cols].mean()

# Find columns with NaN values for test data
test_cols_with_nans = X_test_features_df.columns[X_test_features_df.isna().any()]
test_nan_cols = []
log.info("\nTest data set columns with NaNs:")
for col in test_cols_with_nans:
    test_nan_cols.append(col)
    log.info(f"{col}-{X_test_features_df[col].isna().sum()} NaNs")

# Fill in NaN columns in the test data with means from the training data
X_test_features_df[test_nan_cols] = X_test_features_df[test_nan_cols].fillna(train_means)

# ---------------------------------------------------------------
# Train bloom presence model with cross-validation
# ---------------------------------------------------------------

# Set the folds to be the years
groups = algae_train_meta["vct_year"].values

# Define the cross-validation algorithm as Leave One Group Out
logo_bloom = LeaveOneGroupOut()

# Set regularization alpha for all models, using optimal values from above
reg_alpha = 0.16780683414265007
reg_l1_ratio = 0.00005

# Create the model

# Define a pipeline to impute NaNs, scale the data, and define the model within the cross-validation
# Use ElasticNetCV to handle multiple correlated features better than regular elastic net
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=0.95)),  # keep 95% variance
    ("model", ElasticNetCV(l1_ratio=reg_l1_ratio, fit_intercept=True))
])

# Do cross-validation one year at a time
cv_results = cross_validate(
    pipeline,
    X_lr_train_features,
    y_lr_train_bloom,
    cv=logo_bloom.split(X_lr_train_features, y_lr_train_bloom, groups=groups),
    scoring=['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
    # scoring=["roc_auc", "accuracy", "precision", "recall", "f1"]
)

# Scores for each cross-validation fold:
log.info("\nCross-validation scores for bloom presence model:")
for metric, values in cv_results.items():
    if metric.startswith("test_"):
        display_values = -values.round(4) if "neg_" in metric else values.round(4)
        log.info(f"{metric} {display_values.mean().round(4)} {display_values}")

# Fit the model
pipeline.fit(X_lr_train_features, y_lr_train_bloom)

# Check the R2 score on the training set
lr_bloom_train_r2 = pipeline.score(X_lr_train_features, y_lr_train_bloom)
log.info("\nR2 scores after fitting bloom presence model:")
log.info(f"Training R2: {lr_bloom_train_r2}")

# Get predictions from the model
y_pred_bloom_test = pipeline.predict(X_lr_test_features)

# Check the R2 score on the test set
lr_bloom_test_r2 = pipeline.score(X_lr_test_features, y_lr_test_bloom)
log.info(f"Test R2: {lr_bloom_test_r2}")

# ---------------------------------------------------------------
# Calculate performance metrics for bloom presence model
# ---------------------------------------------------------------

# Plot ROC Curve

# Get out-of-fold predictions (raw regression scores, not probabilities)
y_out_of_fold_scores = cross_val_predict(
    pipeline,
    X_lr_train_features,
    y_lr_train_bloom,
    groups=groups,
    cv=logo_bloom.split(X_lr_train_features, y_lr_train_bloom, groups=groups),
    method="predict" # Because I have a regressor, not a classifier
)

# Calculate classifier metrics using best threshold to maximize F1 score

# Calculate continuous scores
precision, recall, thresholds = precision_recall_curve(
    y_lr_train_bloom,
    y_out_of_fold_scores
)

# Compute F1 scores
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

# Find best threshold
best_idx = np.argmax(f1_scores[:-1])  # exclude last point (no threshold)
best_threshold = thresholds[best_idx]

log.info(f"\nBest threshold (max F1): {best_threshold:.4f}")
log.info(f"Best F1 score: {f1_scores[best_idx]:.4f}")

threshold = best_threshold
y_pred_binary = (y_out_of_fold_scores >= threshold).astype(int)

accuracy = accuracy_score(y_lr_train_bloom, y_pred_binary)
precision = precision_score(y_lr_train_bloom, y_pred_binary)
recall = recall_score(y_lr_train_bloom, y_pred_binary)
f1 = f1_score(y_lr_train_bloom, y_pred_binary)

log.info(f"\nManual classification metrics (threshold={threshold}):")
log.info(f"Accuracy: {accuracy:.3f}")
log.info(f"Precision: {precision:.3f}")
log.info(f"Recall: {recall:.3f}")
log.info(f"F1: {f1:.3f}")

# Draw curves

cv_auc = roc_auc_score(y_lr_train_bloom, y_out_of_fold_scores)
fpr, tpr, _ = roc_curve(y_lr_train_bloom, y_out_of_fold_scores)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"ROC (AUC = {cv_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Linear Regression Bloom Presence ROC Curve")
plt.legend()
plt.grid(True, alpha=0.3)

fig_name = "linear_regression_bloom_presence_roc_curve.png"
fig_path = os.path.join(fig_directory, fig_name)
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
# plt.show()
plt.close()

# Plot Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_lr_train_bloom, y_out_of_fold_scores)

ap = average_precision_score(y_lr_train_bloom, y_out_of_fold_scores)

plt.figure(figsize=(6,6))
plt.plot(recall, precision, label=f"Avg Precision Score = {ap:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Linear Regression Bloom Presence Precision–Recall Curve")
plt.legend()
plt.grid(True)

# Get current limits
x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()

# Compute how far above 1.0 the axis goes
x_pad = x_max - 1
y_pad = y_max - 1

# Set symmetric limits
plt.xlim(-x_pad, 1 + x_pad)
plt.ylim(-y_pad, 1 + y_pad)

fig_name = "linear_regression_bloom_presence_pr_curve.png"
fig_path = os.path.join(fig_directory, fig_name)
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
# plt.show()
plt.close()

# ---------------------------------------------------------------
# Train bloom intensity model with cross-validation
# ---------------------------------------------------------------

# Set the folds to be the years
groups = algae_train_meta["vct_year"].values

# Define the cross-validation algorithm as Leave One Group Out
logo_intensity = LeaveOneGroupOut()

# Create the model
# lr_intensity_model = LinearRegression(fit_intercept=True)
# lr_intensity_model = Lasso(alpha=reg_alpha, fit_intercept=True)
# lr_intensity_model = ElasticNet(alpha=reg_alpha, l1_ratio=reg_l1_ratio, fit_intercept=True)

# Define a pipeline to impute NaNs, scale the data, and define the model
# within the cross-validation
# Use ElasticNetCV to handle multiple correlated features better than regular elastic net
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    # ("pca", PCA(n_components=0.95)),  # keep 95% variance
    ("model", ElasticNetCV(l1_ratio=reg_l1_ratio, fit_intercept=True))
])

# Do cross-validation one year at a time
cv_results = cross_validate(
    pipeline,
    X_lr_train_features,
    y_lr_train_intensity,
    cv=logo_intensity.split(X_lr_train_features, y_lr_train_intensity, groups=groups),
    scoring=['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
    # scoring=["roc_auc", "accuracy", "precision", "recall", "f1"]
)

# Scores for each cross-validation fold:
log.info("\nCross-validation scores for bloom intensity model:")
for metric, values in cv_results.items():
    if metric.startswith("test_"):
        display_values = -values.round(4) if "neg_" in metric else values.round(4)
        log.info(f"{metric} {display_values.mean().round(4)} {display_values}")

# Fit the model
pipeline.fit(X_lr_train_features, y_lr_train_intensity)

# Check coefficient importance
# Convert X_lr_train_features to a DataFrame
X_lr_train_features_df = pd.DataFrame(X_lr_train_features, columns=feature_cols)
# Now plotting works
coef = pipeline.named_steps["model"].coef_
feature_importance = pd.Series(coef, index=X_lr_train_features_df.columns)
feature_importance.sort_values().plot(kind='barh')
plt.title("Linear Regression Bloom Intensity ElasticNet Coefficients")

fig_name = "linear_regression_coefficients.png"
fig_path = os.path.join(fig_directory, fig_name)
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
# plt.show()
plt.close()

# Check the R2 score on the training set
lr_intensity_train_r2 = pipeline.score(X_lr_train_features, y_lr_train_intensity)
log.info("\nR2 scores after fitting bloom presence model:")
log.info(f"Training R2: {lr_intensity_train_r2}")

# Get predictions from the model
y_pred_intensity_test = pipeline.predict(X_lr_test_features)

# Check the R2 score on the test set
lr_intensity_test_r2 = pipeline.score(X_lr_test_features, y_lr_test_intensity)
log.info(f"Test R2: {lr_intensity_test_r2}")

# ---------------------------------------------------------------
# Calculate performance metrics for bloom intensity model
# ---------------------------------------------------------------

# Generate predictions on the test set
y_pred_intensity_test = pipeline.predict(X_lr_test_features)

# True values
y_true_intensity = y_lr_test_intensity

# Evaluation metrics
r2 = r2_score(y_true_intensity, y_pred_intensity_test)
rmse = np.sqrt(mean_squared_error(y_true_intensity, y_pred_intensity_test))
mae = mean_absolute_error(y_true_intensity, y_pred_intensity_test)

log.info(f"\nBloom intensity model performance:")
log.info(f"R²   : {r2:.3f}")
log.info(f"RMSE : {rmse:.3f}")
log.info(f"MAE  : {mae:.3f}")

# Plot residual distribution
residuals = y_true_intensity - y_pred_intensity_test

sns.histplot(residuals, kde=True)
plt.xlabel("Residual")
plt.title("Linear Regression Bloom Intensity Distribution of Residuals")

fig_name = "linear_regression_residual_distro.png"
fig_path = os.path.join(fig_directory, fig_name)
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
# plt.show()
plt.close()
