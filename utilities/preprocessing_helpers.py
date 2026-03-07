import pandas as pd
from datetime import date
from typing import List

MIN_DATE = date(2015, 1, 1)


def load_vct_dataset(src: str = "data/unified_csvs/vct_unified.csv") -> pd.DataFrame:
    """Load the VT Department of Health dataset

    Note that this filters data using a cutoff year and to sites only in Lake Champlain.
    """
    df = pd.read_csv(src)
    df.columns = df.columns.str.lower()
    df = df.rename(columns={
        "reportdate": "report_date",
        "bloomintensity": "bloom_intensity",
        "watertemp": "water_temp",
        "watersurface": "water_surface"
    })
    df["report_date"] = pd.to_datetime(
        df["report_date"], format="%m/%d/%Y"
    ).dt.date
    # Filter to only sites on Lake Champlain.
    # Note that there are some generous spellings of Lake Champlain so this captures all close matches
    df = df[
        df["waterbody"].str.contains("CHA", case=False, na=False)
    ]
    # Filter to dates in our range of interest (data is too sparse prior to this date)
    df = df[df["report_date"] >= MIN_DATE]

    # Clean up region names
    df["region"] = df["region"].apply(lambda region: region.split("-")[-1].strip().title())
    df["region"] = df["region"].replace({
        "South Lake": "Main Lake South",
        "Main Lake": "Main Lake Central",
        "Missiquoi Bay": "Missisquoi Bay",
    })
    return df

def create_lagged_features(df: pd.DataFrame, window_start: int, window_end: int, aggregation_methods: dict) -> pd.DataFrame:
    return df.shift(window_end).rolling(window_start, min_periods=1).agg(
        aggregation_methods
    )

def get_joined_features_and_targets(target_src: str, feature_src: List[str]) -> pd.DataFrame:
    target_df = pd.read_csv(target_src)
    feature_dfs = []
    for src in feature_src:
        feature_dfs.append(
            pd.read_csv(src)
        )
