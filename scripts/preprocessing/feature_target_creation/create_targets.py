"""Transform VT Cyanobacteria Tracker data into a target dataset containing presence of bloom and intensity.

Usage:
    python scripts/preprocessing/feature_target_creation/create_targets.py
"""

import os
import pandas as pd
import numpy as np
from typing import Union

from utilities import data_paths
from utilities.preprocessing_helpers import load_vct_dataset


def create_vct_targets(dst: str) -> pd.DataFrame:
    """Create VCT target data containing bloom presence and intensity and write the outputs to csv.

    Args:
        dst: output file destination. If empty, no output file is saved

    Returns:
        Target dataframe
    """
    vct_df = load_vct_dataset()
    target_df = vct_df[["report_date", "region", "bloom_intensity"]]
    target_df["bloom_intensity"] = target_df["bloom_intensity"].apply(
        _encode_bloom_intensity_as_ordinal
    )
    target_df["bloom"] = target_df["bloom_intensity"] >= 1.0
    target_df = target_df.groupby(["region", "report_date"], as_index=False).agg("max")
    target_df = target_df.dropna().sort_values(["report_date", "region"])

    target_df = target_df.rename(
        columns={
            "report_date": "vct_report_date",
            "region": "vct_region",
            "bloom": "vct_target_bloom",
            "bloom_intensity": "vct_target_bloom_intensity_num",
        }
    )
    if dst is not None:
        target_df.to_csv(dst, index=False)
        os.chmod(
            dst, 0o777
        )  # Open up all the file permissions (read/write/execute for all)
        return target_df


def _encode_bloom_intensity_as_ordinal(bloom_intensity: Union[str, float]) -> float:
    if type(bloom_intensity) == float:
        return bloom_intensity

    if "1a" in bloom_intensity:
        return 0.0
    if "1b" in bloom_intensity:
        return 0.25
    if "1c" in bloom_intensity:
        return 0.5
    if "1d" in bloom_intensity:
        return 0.75
    if "2" in bloom_intensity:
        return 1.0
    if "3" in bloom_intensity:
        return 2.0

    return np.nan


def main():
    create_vct_targets(data_paths.TARGETS_PATH)


if __name__ == "__main__":
    main()
