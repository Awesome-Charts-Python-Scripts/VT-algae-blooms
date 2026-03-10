"""Create VT Department of Environmental Conservation feature dataset.

Transformations occur by averaging all feature measurements over a window and aligning on nearest target location.
The nearest feature site to each target observation was determined through visual observation by plotting target
region and DEC site locations on a map. If multiple DEC site locations are deemed suitable, the missing values
are filled using the next closest suitable site.

Usage:
    python scripts/preprocessing/feature_target_creation/create_dec_features.py
"""

import os
import json
import pandas as pd
from typing import Optional, List
import numpy as np

from utilities import data_paths
from utilities.preprocessing_helpers import create_lagged_features

# Lag window to use for aggregating features
LAG_DAYS = 1
LAG_WINDOW_SIZE = 14

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
    raw_feature_df = _load_raw_dec_data()
    with open(data_paths.VCT_TO_DEC_SITE_MAPPING_PATH, "r") as f:
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
        # In some cases there are multiple DEC sites that map to a single VCT region.
        # We will iteratively fill this dataframe ad do coalesce null entries with the nearest matching site.
        feature_df_for_target_site = pd.DataFrame(
            index=pd.DatetimeIndex(name="date", data=[]), columns=features_to_use
        )
        for matching_dec_site in region_map["DEC sites"]:
            # Subselect the feature rows to match the target site
            feature_rows_for_target_site = pivoted_feature_df[
                pivoted_feature_df["station"] == matching_dec_site
            ].set_index("date")
            # Union the dates across all matching DEC sites for the VCT region
            full_index = sorted(
                set(feature_df_for_target_site.index)
                | set(feature_rows_for_target_site.index)
            )
            # Fill any missing dates in the unioned set
            feature_df_for_target_site = feature_df_for_target_site.reindex(
                full_index
            ).fillna(feature_rows_for_target_site)
        region_feature_dfs.append(
            feature_df_for_target_site.assign(
                region=region_map["VCT site"]
            ).reset_index()
        )

    feature_df = pd.concat(region_feature_dfs)
    # If multiple observations are made on a given date, aggregate them to the daily level sp
    # as not to overweight individual days.
    feature_df = feature_df.groupby(["date", "region"]).agg("mean").reset_index()
    # Lag the features over the window
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


def _load_raw_dec_data():
    dec_df = pd.read_csv(data_paths.DEC_UNIFIED_PATH)
    dec_df.columns = [c.lower() for c in dec_df.columns]
    dec_df["date"] = pd.to_datetime(dec_df["date"], format="%m-%d-%Y")
    # Drop the first 5 characters of the station name which are the numeric code
    dec_df["station"] = dec_df["station"].str[5:]
    return dec_df


def main():
    create_dec_features(FEATURE_COLUMNS_OF_INTEREST, data_paths.DEC_FEATURES_PATH)


if __name__ == "__main__":
    main()
