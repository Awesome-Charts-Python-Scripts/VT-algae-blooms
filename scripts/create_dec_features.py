"""Transform VT Department of Environmental Conservation features so there is one row per target and save to a CSV.

Transformations occur by averaging all feature measurements from the prior 7 day window for each target observation
and aligning on nearest location. The nearest feature location to each target observation is determined using a
euclidean distance. Because the target observations lat/lons change slightly between site readings, an average
lat/lon per site was used.

Usage:
    python scripts/create_dec_features.py \
        -o data/unified_csvs/vt_dec_unified_prepped.csv
"""

import os
import json
import argparse
import pandas as pd
from typing import Optional, List
import numpy as np
from datetime import timedelta

# Input data file paths
TARGET_CSV_PATH = "data/unified_csvs/vct_unified_prepped.csv"
FEATURE_CSV_PATH = "data/unified_csvs/vt_dec.csv"
VCT_TO_DEC_SITE_MAPPING_PATH = "data/data_dictionaries/VCT_to_DEC_site_mappings.json"

# Lookback window to use for aggregating features when pairing with the target observation dates
AGGREGATION_WINDOW_START = timedelta(days=14)
AGGREGATION_WINDOW_END = timedelta(days=1)

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


def create_dec_features(
    features_to_use: List[str] = FEATURE_COLUMNS_OF_INTEREST, dst: Optional[str] = None
) -> pd.DataFrame:
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
    target_df = _get_targets(TARGET_CSV_PATH)
    raw_feature_df = _get_dec_features(FEATURE_CSV_PATH)
    with open(VCT_TO_DEC_SITE_MAPPING_PATH, "r") as f:
        vct_to_dec_site_mapping = json.load(f)

    # The feature dataframe has a row per measurement and we need a row per station and measurement date
    # with the features pivoted as columns. We average by the date here so that we don't overweight
    # measurements that occur multiple times per day.
    pivoted_feature_df = raw_feature_df.pivot_table(
        index=["station", "date"], columns="test", values="result", aggfunc=np.nanmean
    ).sort_index()
    # Standardize the column names with lowercase and subset to the user requested feature list.
    pivoted_feature_df.columns = [c.lower() for c in pivoted_feature_df.columns]
    pivoted_feature_df = pivoted_feature_df[features_to_use]

    # To align our features with our target observations, we create a dataframe from our targets that replicates
    # each row so that we can easily marge on our feature dataframe for the entire window. We then later group
    # by the target observation date to aggregate all feature rows across the window.
    target_df["feature_window"] = target_df["vct_report_date"].apply(
        lambda dt: pd.date_range(
            start=dt - AGGREGATION_WINDOW_START,
            end=dt - AGGREGATION_WINDOW_END,
            inclusive="both",
            freq="D",
        )
    )
    target_df_exploded_by_window = target_df.explode("feature_window").reset_index(
        drop=True
    )

    # Iterate over each target monitoring site and aggregate the features over the interval for each corresponding
    # feature monitoring site. If multiple feature monitoring sites map to the target monitoring site, coalesce
    # the datasets to minimize the number of null entries.
    features_aggregated_weekly = []
    for target_site, grp in target_df_exploded_by_window.groupby("region"):
        # Get the list of matching feature sites
        feature_sites = [
            row["DEC sites"]
            for row in vct_to_dec_site_mapping
            if row["VCT site"] == target_site
        ][0]
        feature_df_for_target_site = pd.DataFrame()
        for matching_dec_site in feature_sites:
            # Subselect the feature rows to match the target site
            feature_rows_for_target_site = pivoted_feature_df.loc[matching_dec_site]
            # Merge with the target dataframe using the window
            target_merged_with_features = grp.merge(
                feature_rows_for_target_site,
                left_on="feature_window",
                right_index=True,
                how="left",
            ).drop(columns=["feature_window"])
            # Aggregate across the window using a mean
            weekly_avg_df = target_merged_with_features.groupby(
                ["region", "vct_report_date"]
            ).mean()
            if feature_df_for_target_site.empty:
                feature_df_for_target_site = weekly_avg_df
                continue
            # If multiple sites can be mapped, coalesce null values
            feature_df_for_target_site = feature_df_for_target_site.fillna(
                weekly_avg_df
            )
        features_aggregated_weekly.append(feature_df_for_target_site)
    all_features = pd.concat(features_aggregated_weekly).reset_index()
    all_features.columns = [f"dec_{c}" for c in all_features.columns]
    if dst is not None:  # Optionally save the results to disk
        all_features.to_csv(dst, index=False)
        os.chmod(
            dst, 0o777
        )  # Open up all the file permissions (read/write/execute for all)
    return all_features


def _get_targets(src: str) -> pd.DataFrame:
    target_df = pd.read_csv(src)
    target_df["vct_report_date"] = pd.to_datetime(
        target_df["vct_report_date"], format="%Y-%m-%d"
    ).dt.date
    return target_df[["region", "vct_report_date"]]


def _get_dec_features(src):
    feature_df = pd.read_csv(src)
    feature_df.columns = [c.lower() for c in feature_df.columns]
    feature_df["date"] = pd.to_datetime(feature_df["date"], format="%m-%d-%Y")
    # Drop the first 5 characters of the station name which are the numeric code
    feature_df["station"] = feature_df["station"].str[5:]
    return feature_df


def main():
    parser = argparse.ArgumentParser(
        description="Generate a CSV of VT DEC features to correspond 1-to-1 with our targets."
    )
    parser.add_argument(
        "-o", action="store", default="out.csv", help="Output file name"
    )
    args = parser.parse_args()
    create_dec_features(dst=args.o)


if __name__ == "__main__":
    main()
