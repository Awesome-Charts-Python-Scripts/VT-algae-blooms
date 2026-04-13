"""Create USGS feature dataset.

Transformations occur by averaging all feature measurements over a window.

Usage:
    python scripts/preprocessing/feature_target_creation/create_usgs_features.py
"""

import os
import pandas as pd
from typing import Optional

from utilities import data_paths
from utilities.preprocessing_helpers import (
    create_lagged_features,
    interpolate_features,
)

# Lag window to use for aggregating features
LAG_DAYS = 1
LAG_WINDOW_SIZE = 7

# Feature columns to store in the final report. These were manually selected according to their feature
# importance and availability of observations.
FEATURE_COLUMNS_OF_INTEREST = [
    "total nitrogen",
    "total phosphorus",
    "dissolved phosphorus",
    "chlorophyll-a",
    "secchi depth",
    "temperature",
]


def create_usgs_features(
    dst: Optional[str] = None, use_lag: bool = True
) -> pd.DataFrame:
    """Create DEC features by aggregating data over a lag window.

    Raw features are stored as rows of single test results across all tests and monitoring sites. To convert
    these into usable features, these must be pivoted so that each test is its own column. The test results
    must then be aggregated across a window so that they can be correctly paired with target observations.

    Args:
        features_to_use: only save features included in this list
        dst: output file destination. If empty, no output file is saved
    Returns:
        Feature dataframe
    """
    # Load the raw data and kip the first row which contains the units for each column
    usgs_raw_df = pd.read_csv(data_paths.USGS_UNIFIED_PATH, index_col=0).iloc[1:]
    usgs_raw_df["datetime"] = pd.to_datetime(
        usgs_raw_df["datetime"], format="%Y-%m-%d"
    ).dt.date

    # Drop the "_cd" code columns and "site_no" column
    usgs_raw_df = usgs_raw_df.drop(
        columns=["site_no"] + [col for col in usgs_raw_df if "_cd" in col]
    )
    # Rename the data value columns by removing first 5 digits + underscore and replacing codes with nicer names
    usgs_cleancols_df = usgs_raw_df.rename(
        columns={
            "datetime": "report_date",
        }
    ).rename(columns={col: _clean_column_name(col) for col in usgs_raw_df.columns})

    # Convert columns to numeric
    for col in usgs_cleancols_df.columns:
        if col == "report_date":
            continue
        usgs_cleancols_df[col] = pd.to_numeric(usgs_cleancols_df[col], errors="coerce")

    # If multiple observations are made on a given date, aggregate them to the daily level sp
    # as not to overweight individual days.
    feature_df = usgs_cleancols_df.groupby("report_date").agg("mean").reset_index()
    # Lag the features over the window
    agg_functions = {
        col: "mean" for col in usgs_cleancols_df.columns if col != "report_date"
    }
    if use_lag:
        feature_df = create_lagged_features(
            feature_df, "report_date", None, LAG_DAYS, LAG_WINDOW_SIZE, agg_functions
        )
    else:
        feature_df = interpolate_features(feature_df, "report_date", None)
    feature_df.columns = [f"usgs_{c}" for c in feature_df.columns]
    if dst is not None:  # Optionally save the results to disk
        feature_df.to_csv(dst, index=False)
        os.chmod(
            dst, 0o777
        )  # Open up all the file permissions (read/write/execute for all)
    return feature_df


def _clean_column_name(col: str) -> str:
    if "_" in col and col.split("_")[0].isdigit():
        parts = col.split("_")
        # Remove first 5 digits part
        col = "_".join(parts[1:])
    return (
        # Turn into lower case and then replace codes with nicer names
        col.lower()
        .replace("00010_", "water_temp_")
        .replace("00095_", "conductivity_")
        .replace("00001", "max")
        .replace("00002", "min")
        .replace("00003", "mean")
    )


def main():
    create_usgs_features(data_paths.USGS_FEATURES_PATH)


if __name__ == "__main__":
    main()
