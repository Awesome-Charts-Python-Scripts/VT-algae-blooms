"""Transform VT Department of Environmental Conservation features so there is one row per target and save to a CSV.

Transformations occur by averaging all feature measurements from the prior 14 day window for each target observation
and aligning on nearest location. The nearest feature location to each target observation is determined using a
euclidean distance. Because the target observations lat/lons change slightly between site readings, an average
lat/lon per site was used.

Usage:
    python scripts/models/random_forest.py
"""

import os
import sklearn
import argparse
import pandas as pd
from datetime import date
from typing import List

# Input data file paths
DATA_CSV_PATH = "data/merged_csvs/algae_merged.csv"
# FEATURE_CSV_PATH = "data/unified_csvs/vt_dec.csv"

TEST_SPLIT_DATE = date(2021, 1, 1)

DEFAULT_FEATURE_LIST = [
  #  "vct_latitude",
  #  "vct_longitude",
  #  "vct_water_temp_trailing",
  #  "vct_water_surface_trailing",  # We should probably convert this to ordinal and do a trailing mean
    "vct_anabaena_trailing",
    "vct_aphanizomenon_trailing",
    "vct_microcystin_trailing",
    "vct_oscillatoria_trailing",
  #  "usgs_water_temp_max_trailing",
  #  "usgs_water_temp_min_trailing",
    "usgs_water_temp_mean_trailing",
  #  "usgs_conductivity_max_trailing",
  #  "usgs_conductivity_min_trailing",
    "usgs_conductivity_mean_trailing",
    "noaa_PRCP",
  #  "noaa_TMAX",
  #  "noaa_TMIN",
    "noaa_AWND",
    "noaa_WSF2",
    "noaa_SNWD",
    "noaa_WDF2",
    "noaa_SNOW",
    "noaa_WSF5",
    "noaa_WDF5",
    "noaa_TAVG",
    "dec_total nitrogen",
    "dec_total phosphorus",
    "dec_dissolved phosphorus",
    "dec_chlorophyll-a",
    "dec_secchi depth",
    "dec_temperature",
    # "vct_target_bloom_intensity",
    # "vct_target_bloom_intensity_num",
]


def create_model(dst: str):
    """Create DEC features per target row through data aggregations.

    Raw features are stored as rows of single test results across all tests and monitoring sites. To convert
    these into usable features, these must be pivoted so that each test is its own column. The test results
    must then be aggregated across a window so that they can be correctly paired with target observations.

    Args:
        features_to_use: only save features included in this list
        dst: output file destination. If empty, no output file is saved
    Returns:
        Feature dataframe
    """
    X, y = _get_features_and_targets("vct_target_bloom", DEFAULT_FEATURE_LIST)
    X_train, X_test, y_train, y_test = _get_train_test_split(X, y)
    groups = [dt.year for dt in y_train.index.get_level_values(0)]
    print("Getting model performance without feature selection")
    results = _run_cross_validation(X_train, y_train, groups)
    features_to_use = _get_important_features(X_train, y_train, groups)
    print(features_to_use)

    print("Getting model performance with feature selection")
    X, y = _get_features_and_targets("vct_target_bloom", features_to_use)
    X_train, X_test, y_train, y_test = _get_train_test_split(X, y)
    results = _run_cross_validation(X_train, y_train, groups)

    # Create groups by year of the data

    # results = sklearn.model_selection.cross_validate(rfc, X_train, y_train, cv=logo, groups=groups)
    #
    # for train, validation in logo.split(X_train, y_train, groups=groups):
    #     X_train_cv = X_train.iloc[train]
    #     y_train_cv = y_train.iloc[train]
    #     X_validation_cv = X_train.iloc[validation]
    #     y_validation_cv = y_train.iloc[validation]

def _get_features_and_targets(target: str, feature_list: List[str]):
    df = pd.read_csv(DATA_CSV_PATH).infer_objects()
    df["vct_report_date"] = pd.to_datetime(df["vct_report_date"]).dt.date
    df["dec_temperature"] = df["dec_temperature"].fillna(
        df["usgs_water_temp_mean_trailing"]
    )
    index_cols = ["vct_report_date", "vct_region"]
    df = df[index_cols + feature_list + [target]].dropna().set_index(index_cols).sort_index()
    X = df.drop(columns=target)
    y = df[target]
    return X, y


def _get_train_test_split(X, y):
    X_train = X[X.index.get_level_values(0) < TEST_SPLIT_DATE]
    X_test = X[X.index.get_level_values(0) >= TEST_SPLIT_DATE]
    y_train = y[y.index.get_level_values(0) < TEST_SPLIT_DATE]
    y_test = y[y.index.get_level_values(0) >= TEST_SPLIT_DATE]
    return (
        X_train, X_test, y_train, y_test
    )
def _run_cross_validation(X, y, groups, n_estimators = 1000):
    logo = sklearn.model_selection.LeaveOneGroupOut()
    rfc = sklearn.ensemble.RandomForestClassifier(n_estimators=n_estimators, max_features='sqrt',
                                                  random_state=42)
    results = sklearn.model_selection.cross_validate(rfc, X, y, cv=logo, groups=groups)
    print(results["test_score"])
    return results

def _get_important_features(X, y, groups, n_estimators=1000, threshold=0.01):
    logo = sklearn.model_selection.LeaveOneGroupOut()
    important_features = {}
    for i, (train, test) in enumerate(logo.split(X, y, groups=groups)):
        rfc = sklearn.ensemble.RandomForestClassifier(n_estimators=n_estimators, max_features='sqrt',
                                                      random_state=i)
        X_train = X.iloc[train]
        y_train = y.iloc[train]
        X_test = X.iloc[test]
        y_test = y.iloc[test]
        rfc.fit(X_train, y_train)
        y_pred = rfc.predict(X_test)
        score = sklearn.metrics.accuracy_score(y_test, y_pred)
        for feature, importance in zip(X_train.columns, rfc.feature_importances_):
            if importance < threshold:
                continue
            important_features[feature] = important_features.get(feature, []) + [importance]
        print(score)
    return [feat for feat, importances in important_features.items() if len(importances) == len(set(groups))]


def main():
    parser = argparse.ArgumentParser(
        description="Generate a CSV of VT DEC features to correspond 1-to-1 with our targets."
    )
    parser.add_argument(
        "-o", action="store", default="out.csv", help="Output file name"
    )
    args = parser.parse_args()
    create_model(dst=args.o)


if __name__ == "__main__":
    main()
