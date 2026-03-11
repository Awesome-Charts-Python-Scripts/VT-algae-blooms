"""Create NOAA feature dataset.

Transformations occur by averaging all feature measurements over a window.

Usage:
    python scripts/preprocessing/feature_target_creation/create_noaa_features.py
"""

import os
import pandas as pd
from typing import Optional

from utilities import data_paths
from utilities.preprocessing_helpers import create_lagged_features

# Lag window to use for aggregating features
LAG_DAYS = 1
LAG_WINDOW_SIZE = 7


def create_noaa_features(dst: Optional[str] = None) -> pd.DataFrame:
    """Create NOAA features by aggregating data over a lag window.

    Args:
        dst: output file destination. If empty, no output file is saved
    Returns:
        Feature dataframe
    """
    # Load CSV
    df = pd.read_csv(data_paths.NOAA_UNIFIED_PATH)

    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.set_index("DATE")
    df = df.loc["2012-01-01":"2024-12-31"]  # Time range is 2012-2024

    # Rename columns to more descriptive names
    df = df.rename(
        columns={
            "STATION": "station",
            "NAME": "station_name",
            "PRCP": "precipitation",
            "SNOW": "snowfall",
            "SNWD": "snow_depth",
            "PSUN": "sunshine_percent",
            "TSUN": "sunshine_total",
            "TAVG": "air_temp_mean",
            "TMAX": "air_temp_max",
            "TMIN": "air_temp_min",
            "WESD": "water_on_ground",
            "AWND": "wind_speed_mean",
            "FMTM": "wind_fastest_time",
            "PGTM": "peak_gust_time",
            "WDF2": "wind_direction_2_min",
            "WDF5": "wind_direction_5_min",
            "WSF2": "wind_speed_2_min",
            "WSF5": "wind_speed_5_min",
            "WT01": "fog",
            "WT02": "heavy_fog",
            "WT03": "thunder",
            "WT04": "sleet_hail",
            "WT05": "hail",
            "WT06": "glaze_rime",
            "WT07": "dust_ash",
            "WT08": "smoke_haze",
            "WT09": "drifting_snow",
            "WT11": "high_winds",
            "WT13": "mist",
            "WT14": "drizzle",
            "WT15": "freezing_drizzle",
            "WT16": "rain",
            "WT17": "freezing_rain",
            "WT18": "snow_pellets",
            "WT19": "unknown_precipitation",
            "WT21": "ground_fog",
            "WT22": "ice_fog",
            "WV01": "fog_vicinity",
            "WV03": "thunder_vicinity",
        }
    )

    # Sort columns by non-null values
    df_numeric = df.select_dtypes(include="number")

    coverage = pd.DataFrame(
        {
            "non_null_count": df_numeric.notna().sum(),
            "non_null_percentage": df_numeric.notna().mean(),
        }
    ).sort_values("non_null_percentage", ascending=False)

    # Drop features below threshold
    THRESHOLD = 0.8  # Minimum data coverage to be a usable feature

    usable_features = coverage.loc[
        coverage["non_null_percentage"] >= THRESHOLD
    ].index.tolist()
    df_numeric = df_numeric[usable_features]

    # If multiple observations are made on a given date, aggregate them to the daily level sp
    # as not to overweight individual days.
    df_numeric = df_numeric.groupby("DATE").agg("mean").reset_index()
    # Lag the features over the window
    agg_functions = {col: "mean" for col in df_numeric.columns if col != "DATE"}
    feature_df = create_lagged_features(
        df_numeric, "DATE", None, LAG_DAYS, LAG_WINDOW_SIZE, agg_functions
    )
    feature_df = feature_df.rename(columns=lambda c: f"noaa_{c}")

    # Export averages to CSV
    if dst is not None:  # Optionally save the results to disk
        feature_df.to_csv(dst, index=False)
        os.chmod(
            dst, 0o777
        )  # Open up all the file permissions (read/write/execute for all)
    return feature_df


def main():
    create_noaa_features(data_paths.NOAA_FEATURES_PATH)


if __name__ == "__main__":
    main()
