"""Transform VT Department of Environmental Conservation features so there is onw row per target and save to a CSV.

Transformations occur by averaging all feature measurements from the prior 7 day window for each target observation
and aligning on nearest location. The nearest feature location to each target observation is determined using a
euclidean distance. Because the target observations lat/lons change slightly between site readings, an average
lat/lon per site was used.

Usage:
    python scripts/create_dec_features.py \
        -o data/unified_csvs/vt_dec_unified_prepped.csv
"""

import os
import argparse
import pandas as pd
from typing import Dict
from functools import partial
import numpy as np
from datetime import timedelta
import warnings

# Suppress noisy warnings
warnings.filterwarnings("ignore", message=".*Mean of empty slice.*")


TARGET_CSV_PATH = "data/unified_csvs/vct_unified.csv"
FEATURE_CSV_PATH = "data/unified_csvs/vt_dec.csv"
DEC_STATION_LOCATIONS_CSV = "data/data_dictionaries/DEC_station_locations.csv"
AGGREGATION_WINDOW_START = timedelta(days=7)
AGGREGATION_WINDOW_END = timedelta(days=1)


def create_dec_features(dst: str) -> None:
    """Create DEC features per target row through data aggregations."""

    def aggregate_features_weekly(
        station_mapping: Dict[str, str],
        pivoted_features: pd.DataFrame,
        target_row: pd.Series,
    ) -> pd.Series:
        """Use weekly interval prior to each target observation and  average reading for each feature."""
        target_report_date = target_row["REPORTDATE"]
        target_station = target_row["STATION"]
        # If the report date or station is invalid, return an empty row
        if pd.isnull(target_report_date) or pd.isnull(target_station):
            return pd.Series(index=pivoted_features.columns, data=np.nan)
        # Generate weekly averages of each feature
        nearest_dec_station = station_mapping[target_station]
        window_start = str(target_report_date - AGGREGATION_WINDOW_START)
        window_end = str(target_report_date - AGGREGATION_WINDOW_END)
        # Use np.nanmean to ignore nan values when generating averages
        weekly_averages = np.nanmean(
            pivoted_features.loc[
                nearest_dec_station, window_start:window_end, :
            ].values,
            axis=0,
        )
        return pd.Series(index=pivoted_features.columns, data=weekly_averages)

    target_df = _get_targets(TARGET_CSV_PATH)
    raw_feature_df = _get_dec_features(FEATURE_CSV_PATH)
    dec_site_locations_df = pd.read_csv(DEC_STATION_LOCATIONS_CSV)
    vct_to_dec_station_mapping = _map_vct_stations_to_nearest_dec_station(
        target_df, dec_site_locations_df
    )

    # The feature dataframe has a row per measurement and we need a row per station and measurement date
    # with the features pivoted as columns. We average by the date here so that we don't overweight
    # measurements that occur multiple times per day.
    # TODO: Figure out if we should be aggregate tests across all depths and stratum
    # raw_feature_df["Test"] = raw_feature_df.apply(lambda row: f"{row['Test']}_{row['Stratum']}_{row['Depth']}", axis=1)
    pivoted_feature_df = raw_feature_df.pivot_table(
        index=["Station", "Date"], columns="Test", values="Result", aggfunc=np.nanmean
    )
    # Aggregate the features into weekly windows by station
    aggregated_features = target_df[["STATION", "REPORTDATE"]].apply(
        partial(
            aggregate_features_weekly, vct_to_dec_station_mapping, pivoted_feature_df
        ),
        axis=1,
    )
    aggregated_features.to_csv(dst, index=False)
    os.chmod(
        dst, 0o777
    )  # Open up all the file permissions (read/write/execute for all)


def _get_targets(src: str) -> pd.DataFrame:
    target_df = pd.read_csv(src).astype({"LATITUDE": float, "LONGITUDE": float})
    target_df["REPORTDATE"] = pd.to_datetime(
        target_df["REPORTDATE"], format="%m/%d/%Y"
    ).dt.date
    return target_df


def _get_dec_features(src):
    feature_df = pd.read_csv(src)
    feature_df["Date"] = pd.to_datetime(feature_df["Date"], format="%m-%d-%Y")
    return feature_df


def _map_vct_stations_to_nearest_dec_station(
    target_df: pd.DataFrame, dec_site_locations_df: pd.DataFrame
) -> Dict[str, str]:
    """Generate a dictionary mapping of target stations to DEC stations using a euclidean distance"""

    def get_nearest_dec_station(dec_station_locations, target_row):
        distances = (
            (dec_station_locations["degrees_latitude"] - target_row["LATITUDE"]) ** 2
            + (dec_station_locations["degrees_longitude"] - target_row["LONGITUDE"])
            ** 2
        ) ** (1 / 2)
        return dec_station_locations.loc[distances.idxmin()]["DEC_station"]

    target_df_stations = (
        target_df[["STATION", "LATITUDE", "LONGITUDE"]]
        .groupby("STATION")
        .agg("mean")
        .reset_index()
    )
    target_df_stations = target_df_stations.drop_duplicates()
    target_df_stations["NEAREST_DEC_STATION"] = target_df_stations.apply(
        partial(get_nearest_dec_station, dec_site_locations_df), axis=1
    )
    return dict(
        zip(target_df_stations["STATION"], target_df_stations["NEAREST_DEC_STATION"])
    )


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
