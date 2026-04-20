"""
Author: Josh Fishbein
Creates a baseline prediction of the most common bloom status for a given day of the year by site location. For
example if St Albans Bay was observed to have a bloom on August 12th for 4 out of the training years, then a bloom
will be predicted for St Albans Bay for the test period. If there are an equal number of bloom and no bloom events
for a given day/location, then no bloom is selected.

Usage:
    python scripts/baseline/baseline2.py
"""

import sklearn
import pandas as pd
from pprint import pprint
from datetime import date

from utilities.preprocessing_helpers import (
    get_joined_features_and_targets,
    get_train_test_split,
)

TARGET = "vct_target_bloom"


def create_model():
    """Baseline model that always predicts "no bloom"""
    df = get_joined_features_and_targets().set_index(["vct_region", "vct_report_date"])
    X = df.drop(columns=TARGET)
    y = df[[TARGET]]
    _, _, y_train, y_test = get_train_test_split(X, y)

    full_index = pd.date_range(
        date(y_train.index.get_level_values("vct_report_date").min().year, 1, 1),
        date(y_train.index.get_level_values("vct_report_date").max().year + 1, 1, 1),
        freq="D",
    )
    all_dates_by_region = []
    for region, grp in y_train.groupby("vct_region"):
        grp = grp.reset_index()
        grp["vct_report_date"] = pd.to_datetime(grp["vct_report_date"])
        grp = (
            grp.set_index("vct_report_date")
            .reindex(full_index, method="nearest")
            .ffill()
            .bfill()
        )
        grp.index.names = ["vct_report_date"]
        all_dates_by_region.append(
            grp.reset_index().set_index(["vct_region", "vct_report_date"])
        )

    y_train = pd.concat(all_dates_by_region)
    y_train["date_of_year"] = y_train.index.get_level_values(
        "vct_report_date"
    ).strftime("%m-%d")
    y_pred_single_year = (
        y_train.reset_index()
        .groupby(["vct_region", "date_of_year"])
        .agg({TARGET: lambda x: x.mode().iloc[0]})
    )

    y_pred_all_test_years = []
    y_test_years = set(
        pd.to_datetime(y_test.index.get_level_values("vct_report_date")).strftime("%Y")
    )
    for year in y_test_years:
        y_pred_single_year["vct_report_date"] = pd.to_datetime(
            [
                f"{year}-{dt}"
                for dt in y_pred_single_year.index.get_level_values("date_of_year")
            ],
            errors="coerce",
        )
        y_pred_all_test_years.append(
            y_pred_single_year.reset_index().set_index(
                ["vct_region", "vct_report_date"]
            )
        )

    y_pred = pd.concat(y_pred_all_test_years)
    y_pred = y_pred.loc[y_test.index][TARGET]
    metrics = {
        "roc_auc": sklearn.metrics.roc_auc_score(y_test.values, y_pred.values),
        "accuracy": sklearn.metrics.accuracy_score(y_test.values, y_pred.values),
        "precision": sklearn.metrics.precision_score(y_test.values, y_pred.values),
        "recall": sklearn.metrics.recall_score(y_test.values, y_pred.values),
        "f1": sklearn.metrics.f1_score(y_test.values, y_pred.values),
    }

    pprint(metrics, indent=4, sort_dicts=False)


if __name__ == "__main__":
    create_model()
