import pandas as pd
from datetime import date
from typing import List, Optional, Tuple

from utilities import data_paths

MIN_DATE = date(2015, 1, 1)
MAX_DATE = date(2022, 12, 31)
MONTH_START = 5
MONTH_END = 12
TEST_SPLIT_DATE = date(2021, 1, 1)


def load_vct_dataset() -> pd.DataFrame:
    """Load the VT Department of Health dataset

    Note that this filters data using a cutoff year and to sites only in Lake Champlain.
    """
    df = pd.read_csv(data_paths.VCT_UNIFIED_PATH)
    df.columns = df.columns.str.lower()
    df = df.rename(
        columns={
            "reportdate": "report_date",
            "bloomintensity": "bloom_intensity",
            "watertemp": "water_temp",
            "watersurface": "water_surface",
        }
    )
    df["report_date"] = pd.to_datetime(df["report_date"], format="%m/%d/%Y").dt.date
    # Filter to only sites on Lake Champlain.
    # Note that there are some generous spellings of Lake Champlain so this captures all close matches
    df = df[df["waterbody"].str.contains("CHA", case=False, na=False)]
    # Filter to dates in our range of interest (data is too sparse prior to this date)
    df = df[df["report_date"] >= MIN_DATE]

    # Clean up region names
    df["region"] = df["region"].apply(
        lambda region: region.split("-")[-1].strip().title()
    )
    df["region"] = df["region"].replace(
        {
            "South Lake": "Main Lake South",
            "Main Lake": "Main Lake Central",
            "Missiquoi Bay": "Missisquoi Bay",
        }
    )
    return df


def create_lagged_features(
    df: pd.DataFrame,
    date_col: str,
    region_col: Optional[str],
    lag_days: int,
    lag_window_size: int,
    aggregation_methods: dict,
) -> pd.DataFrame:
    def _lag_features_over_window(
        df_to_lag: pd.DataFrame, shift: int, window_size: int, agg_methods: dict
    ) -> pd.DataFrame:
        df_to_lag = df_to_lag.set_index(date_col)
        min_date = min((MIN_DATE, pd.to_datetime(df_to_lag.index.min()).date()))
        max_date = max((MAX_DATE, pd.to_datetime(df_to_lag.index.max()).date()))
        full_index = pd.date_range(min_date, max_date, freq="D")
        df_to_lag = df_to_lag.reindex(full_index)
        df_to_lag = (
            df_to_lag.shift(periods=shift)
            .rolling(window_size, min_periods=1)
            .agg(agg_methods)
        )
        return df_to_lag.reset_index(names=date_col)

    original_cols = df.columns
    if region_col is None:
        return _lag_features_over_window(
            df, lag_days, lag_window_size, aggregation_methods
        )[original_cols]

    all_dfs = []
    for region, region_df in df.groupby(region_col):
        lagged_df = _lag_features_over_window(
            region_df, lag_days, lag_window_size, aggregation_methods
        ).assign(**{region_col: region})
        all_dfs.append(lagged_df)
    return pd.concat(all_dfs)[original_cols]


def interpolate_features(
    df: pd.DataFrame,
    date_col: str,
    region_col: Optional[str],
) -> pd.DataFrame:
    full_index = pd.date_range(MIN_DATE, MAX_DATE, freq="D")
    if region_col is None:
        return (
            df.set_index(date_col)
            .reindex(full_index)
            .interpolate(method="nearest")
            .ffill()
            .bfill()
            .reset_index(names=date_col)
        )

    all_region_dfs = []
    for region, region_df in df.groupby(region_col):
        all_region_dfs.append(
            region_df.set_index(date_col)
            .reindex(full_index)
            .drop(columns=[region_col])
            .astype(float)
            .interpolate(method="nearest")
            .ffill()
            .bfill()
            .assign(**{region_col: region})
        )
    return pd.concat(all_region_dfs).sort_index().reset_index(names=date_col)


def get_joined_features_and_targets(
    target_src: str = data_paths.TARGETS_PATH,
    feature_src: List[str] = [
        data_paths.DEC_FEATURES_PATH,
        data_paths.NOAA_FEATURES_PATH,
        data_paths.USGS_FEATURES_PATH,
        data_paths.VCT_FEATURES_PATH,
    ],
) -> pd.DataFrame:
    full_df = pd.read_csv(target_src)
    for src in feature_src:
        feature_df = pd.read_csv(src)
        date_cols = [col for col in feature_df.columns if "date" in col.lower()]
        region_cols = [col for col in feature_df.columns if "region" in col.lower()]
        assert (
            len(date_cols) == 1
        ), f"Feature dataframe only supports one date column, found {len(date_cols)}"
        assert (
            len(region_cols) <= 1
        ), f"Feature dataframe only supports zero or one region column, found {len(region_cols)}"
        feature_merge_cols = (
            [date_cols[0], region_cols[0]] if region_cols else [date_cols[0]]
        )
        target_merge_cols = (
            ["vct_report_date", "vct_region"] if region_cols else ["vct_report_date"]
        )
        full_df = full_df.merge(
            feature_df,
            left_on=target_merge_cols,
            right_on=feature_merge_cols,
            how="inner",
        ).drop(columns=list(set(feature_merge_cols) - set(target_merge_cols)))
    full_df["vct_report_date"] = pd.to_datetime(full_df["vct_report_date"]).dt.date
    return full_df


def get_train_test_split(
    X: pd.DataFrame, y: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_train = X[X.index.get_level_values("vct_report_date") < TEST_SPLIT_DATE]
    X_test = X[X.index.get_level_values("vct_report_date") >= TEST_SPLIT_DATE]
    y_train = y[y.index.get_level_values("vct_report_date") < TEST_SPLIT_DATE]
    y_test = y[y.index.get_level_values("vct_report_date") >= TEST_SPLIT_DATE]
    return X_train, X_test, y_train, y_test
