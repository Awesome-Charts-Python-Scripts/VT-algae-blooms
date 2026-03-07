"""Transform VT Department of Health data into a feature dataset by aggregating features over a 14-day window

Usage:
    python scripts/create_vct_features.py \
        -o data/unified_csvs/vct_features.csv
"""

import os
import argparse
import pandas as pd
import numpy as np
from typing import Union

from utilities.preprocessing_helpers import load_vct_dataset, create_lagged_features

# Input data file paths
SRC_CSV_PATH = "data/unified_csvs/vct_unified.csv"
LAG_WINDOW_START = 14
LAG_WINDOW_END = 1

def create_vct_targets(dst: str) -> pd.DataFrame:
    """Create VCT feature data by lagging data over a 14-day window.

    Args:
        dst: output file destination
    Returns:
        Feature dataframe
    """
    vct_df = load_vct_dataset(SRC_CSV_PATH)

    # One-hot encode the most observed cyanotaxa types
    vct_df = _one_hot_encode_cynotaxa(vct_df)
    # Encode the water surfaces as ordinal values
    vct_df["water_surface"] = vct_df["water_surface"].apply(_encode_water_surface_as_ordinal)


    # Drop columns with insufficient data or that do not contain relevant feature information
    vct_df = vct_df.drop(columns=[
        "municipality",
        "reporttime",
        "reportfrequency",
        "affiliation",
        "details",
        "webstatus",
        "bloom_intensity",
        "anatoxin",  # same with the other speces columns
        "othertaxa",
    ])

    # Replace out of range values
    vct_df.loc[pd.to_numeric(vct_df["water_temp"], errors="coerce") > 100, "water_temp"] = np.nan

    # Aggregate values to daily
    aggregation_methods = {
        "water_surface": "mean",
        "water_temp": "mean",
        "anabaena": "max",
        "aphanizomenon": "max",
        "microcystin": "max",
        "oscillatoria": "max",
    }
    for col in aggregation_methods.keys():
        vct_df[col] = pd.to_numeric(vct_df[col], errors="coerce")
    vct_df = vct_df.groupby(["report_date", "region"]).agg(aggregation_methods)

    # Lag features over a window
    vct_df = create_lagged_features(vct_df, LAG_WINDOW_START, LAG_WINDOW_END, aggregation_methods).reset_index()
    vct_df.columns = [f"vct_{col}" for col in vct_df.columns]
    vct_df.to_csv(dst, index=False)
    os.chmod(
        dst, 0o777
    )  # Open up all the file permissions (read/write/execute for all)
    return vct_df


def _one_hot_encode_cynotaxa(vct_df: pd.DataFrame) -> pd.DataFrame:
    cyanotaxa_map = {
        "anabaena": "anabaen",
        "aphanizomenon": "aphanizomen",
        "microcystin": "microcyst",
        "oscillatoria": "oscillator"
    }
    for col_name, pattern in cyanotaxa_map.items():
        vct_df[col_name] = vct_df["cyanotaxa"].str.contains(pattern, case=False, na=False).astype(int)
    return vct_df.drop(columns=["cyanotaxa"])

def _encode_water_surface_as_ordinal(water_surface: str) -> float:
    if type(water_surface) == float:
        return water_surface

    if water_surface.lower() == "calm":
        return 0.0
    if water_surface.lower() == "rolling":
        return 1.0
    if water_surface.lower() == "white caps":
        return 2.0
    return np.nan

def main():
    parser = argparse.ArgumentParser(
        description="Generate a CSV of VT VCT features to correspond with our targets."
    )
    parser.add_argument(
        "-o", action="store", default="out.csv", help="Output file name"
    )
    args = parser.parse_args()
    create_vct_targets(dst=args.o)


if __name__ == "__main__":
    main()
