"""Transform VT Department of Environmental Conservation features so there is one row per target and save to a CSV.

Transformations occur by averaging all feature measurements from the prior 7 day window for each target observation
and aligning on nearest location. The nearest feature location to each target observation is determined using a
euclidean distance. Because the target observations lat/lons change slightly between site readings, an average
lat/lon per site was used.

Usage:
    python scripts/create_dec_features.py \
        -o data/unified_csvs/dec_features.csv
"""

import os
import json
import argparse
import pandas as pd
from typing import Optional, List
import numpy as np

from utilities.preprocessing_helpers import create_lagged_features

# Input data file paths
FEATURE_CSV_PATH = "data/unified_csvs/vt_dec.csv"
VCT_TO_DEC_SITE_MAPPING_PATH = "data/data_dictionaries/VCT_to_DEC_site_mappings.json"

# Lookback window to use for aggregating features when pairing with the target observation dates
LAG_WINDOW_SIZE = 14
LAG_DAYS = 1

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
    raw_feature_df = _get_dec_features(FEATURE_CSV_PATH)
    with open(VCT_TO_DEC_SITE_MAPPING_PATH, "r") as f:
        vct_to_dec_site_mapping = json.load(f)

    # The feature dataframe has a row per measurement and we need a row per station and measurement date
    # with the features pivoted as columns. We average by the date here so that we don't overweight
    # measurements that occur multiple times per day.
    pivoted_feature_df = raw_feature_df.pivot_table(
        index=["station", "date"], columns="test", values="result", aggfunc=np.nanmean
    )
    # Standardize the column names with lowercase and subset to the user requested feature list.
    pivoted_feature_df.columns = [c.lower() for c in pivoted_feature_df.columns]
    pivoted_feature_df = pivoted_feature_df[features_to_use].reset_index()

    region_feature_dfs = []
    for region_map in vct_to_dec_site_mapping:
        feature_df_for_target_site = pd.DataFrame(
            index=pd.DatetimeIndex(name="date", data=[]), columns=features_to_use
        )
        for matching_dec_site in region_map["DEC sites"]:
            # Subselect the feature rows to match the target site
            feature_rows_for_target_site = pivoted_feature_df[
                pivoted_feature_df["station"] == matching_dec_site
            ].set_index("date")
            full_index = sorted(
                set(feature_df_for_target_site.index)
                | set(feature_rows_for_target_site.index)
            )
            feature_df_for_target_site = feature_df_for_target_site.reindex(
                full_index
            ).fillna(feature_rows_for_target_site)
        region_feature_dfs.append(
            feature_df_for_target_site.assign(
                region=region_map["VCT site"]
            ).reset_index()
        )

    feature_df = pd.concat(region_feature_dfs)
    agg_functions = {col: "mean" for col in FEATURE_COLUMNS_OF_INTEREST}
    feature_df = create_lagged_features(
        feature_df, "date", "region", LAG_DAYS, LAG_WINDOW_SIZE, agg_functions
    )
    feature_df = feature_df[["date", "region", *FEATURE_COLUMNS_OF_INTEREST]]
    feature_df.columns = [f"dec_{c}" for c in feature_df.columns]
    if dst is not None:  # Optionally save the results to disk
        feature_df.to_csv(dst, index=False)
        os.chmod(
            dst, 0o777
        )  # Open up all the file permissions (read/write/execute for all)
    return feature_df


def _get_targets(src: str) -> pd.DataFrame:
    target_df = pd.read_csv(src)
    target_df["vct_report_date"] = pd.to_datetime(
        target_df["vct_report_date"], format="%Y-%m-%d"
    ).dt.date
    return target_df[["vct_region", "vct_report_date"]]


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
