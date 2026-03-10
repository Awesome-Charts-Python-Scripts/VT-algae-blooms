"""Creates a random forest ensemble model and recommends most important features for a model.

Recommended most important features are:
[
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

import sklearn
import argparse
from datetime import date
from typing import List

from utilities.preprocessing_helpers import get_joined_features_and_targets

TEST_SPLIT_DATE = date(2021, 1, 1)

# Comment out features that are highly correlated with other features
DEFAULT_FEATURE_LIST = [
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
    results = _run_cross_validation(X_train, y_train, groups)
    print(f"Model performance without feature selection: {results['test_score']}")

    important_features = _get_important_features(X_train, y_train, groups)
    X, y = _get_features_and_targets("vct_target_bloom", important_features)
    X_train, X_test, y_train, y_test = _get_train_test_split(X, y)
    results = _run_cross_validation(X_train, y_train, groups)
    print(f"Model performance with feature selection: {results['test_score']}")
    print(f"Recommended features to use: {important_features}")


def _get_features_and_targets(target: str, feature_list: List[str]):
    df = get_joined_features_and_targets()
    df["dec_temperature"] = df["dec_temperature"].fillna(
        df["usgs_water_temp_mean"]
    )
    index_cols = ["vct_report_date", "vct_region"]
    df = (
        df[index_cols + feature_list + [target]]
        .dropna()
        .set_index(index_cols)
        .sort_index()
    )
    X = df.drop(columns=target)
    y = df[target]
    return X, y


def _get_train_test_split(X, y):
    X_train = X[X.index.get_level_values(0) < TEST_SPLIT_DATE]
    X_test = X[X.index.get_level_values(0) >= TEST_SPLIT_DATE]
    y_train = y[y.index.get_level_values(0) < TEST_SPLIT_DATE]
    y_test = y[y.index.get_level_values(0) >= TEST_SPLIT_DATE]
    return (X_train, X_test, y_train, y_test)


def _run_cross_validation(X, y, groups, n_estimators=1000):
    logo = sklearn.model_selection.LeaveOneGroupOut()
    rfc = sklearn.ensemble.RandomForestClassifier(
        n_estimators=n_estimators, max_features="sqrt", random_state=42
    )
    results = sklearn.model_selection.cross_validate(rfc, X, y, cv=logo, groups=groups)
    return results


def _get_important_features(X, y, groups, n_estimators=1000, threshold=0.01):
    logo = sklearn.model_selection.LeaveOneGroupOut()
    important_features = {}
    for i, (train, test) in enumerate(logo.split(X, y, groups=groups)):
        rfc = sklearn.ensemble.RandomForestClassifier(
            n_estimators=n_estimators, max_features="sqrt", random_state=i
        )
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
        description="Generate a CSV of VT DEC features to correspond 1-to-1 with our targets."
    )
    parser.add_argument(
        "-o", action="store", default="out.csv", help="Output file name"
    )
    args = parser.parse_args()
    create_model(dst=args.o)


if __name__ == "__main__":
    main()
