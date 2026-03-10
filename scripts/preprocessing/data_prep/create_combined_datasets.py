"""Copy and combine raw datasets so that they can be preprocessed for feature and target creation.

Usage:
    python scripts/preprocessing/data_prep/create_combined_datasets.py
"""

import io
import os
import shutil
import zipfile
import pandas as pd
from typing import List, Callable

from utilities import data_paths


def create_unified_csv_datasets():
    shutil.copyfile(data_paths.USGS_RAW_PATH, data_paths.USGS_UNIFIED_PATH)
    shutil.copyfile(data_paths.NOAA_RAW_PATH, data_paths.NOAA_UNIFIED_PATH)
    _read_zipfile_files_into_combined_df(
        data_paths.VCT_RAW_PATH, _read_vct_excel
    ).to_csv(data_paths.VCT_UNIFIED_PATH, index=False)
    _read_zipfile_files_into_combined_df(
        data_paths.DEC_RAW_PATH, _read_html_table
    ).to_csv(data_paths.DEC_UNIFIED_PATH, index=False)

    # Open up all the file permissions (read/write/execute for all)
    for unified_csv in os.listdir(data_paths.UNIFIED_CSVS_DIR):
        os.chmod(os.path.join(data_paths.UNIFIED_CSVS_DIR, unified_csv), 0o777)


def _read_html_table(content: io.BytesIO) -> pd.DataFrame:
    return pd.read_html(content)[0]


def _read_vct_excel(content: io.BytesIO) -> pd.DataFrame:
    filename = os.path.split(content.name)[-1]
    # Choose excel sheet based on the mapping
    sheet_map = {
        "ENV_EPHT-cyanobacteria-season-summary-2012.xls": 0,
        "ENV_EPHT-cyanobacteria-season-summary-2013.xls": 0,
        "ENV_EPHT-cyanobacteria-season-summary-2014.xls": 0,
        "ENV_EPHT-cyanobacteria-season-summary-2015.xls": 0,
        "ENV_EPHT-cyanobacteria-season-summary-2016.xlsx": 1,
        "ENV_EPHT-cyanobacteria-season-summary-2017.xlsx": 1,
        "ENV_EPHT-cyanobacteria-season-summary-2018.xlsx": 1,
        "ENV_EPHT-cyanobacteria-season-summary-2019.xlsx": 0,
        "ENV_EPHT-cyanobacteria-season-summary-2020.xlsx": 0,
        "env-epht-cyanobacteria-season-summary-2021.xlsx": 0,
        "env-epht-cyanobacteria-season-summary-2022.xlsx": 0,
    }
    df = pd.read_excel(content, sheet_name=sheet_map[filename])
    return _standardize_vct_df_columns(filename, df)


def _standardize_vct_df_columns(filename: str, df: pd.DataFrame) -> pd.DataFrame:
    """VCT columns vary slightly across reports so standardize as follows:

    - Convert to uppercase
    - Remove spaces
    - Remove underscores
    - Remove forward slashes
    - remove anything from UGL onward
    - standardize latitude ad longitude columns
    - other text changes in clean_col section below
    """
    new_columns = []
    for col in df.columns:
        # basic cleaning
        clean_col = (
            col.upper()
            .replace(" ", "")
            .replace("_", "")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "")
            .replace("PRESENT", "")
            .replace("24HR", "")
            .replace("H2O", "WATER")
            .replace("SAMPLEDATE", "REPORTDATE")
            .replace("POTENTIALLYTOXICCYANOBACTERIA", "CYANOTAXA")
            .replace("BLOOMINTENSITYALL", "BLOOMINTENSITY")
            .replace("SITENAME", "STATION")
            .replace("SITE#2015", "SITE")
            .replace("SITENUMBER", "SITE")
            .replace("SAMPLING", "SAMPLE")
            .replace("LAKE", "WATERBODY")
            .replace("STATUSWEB", "WEBSTATUS")
            .replace("ANOTOXIN", "ANATOXIN")
            .replace("OTHERALGAEANDNON-TOXICCYANOBACTERIA", "OTHERTAXA")
            .replace("⁰F", "")
            .replace("TEMPERATURE", "TEMP")
        )
        # remove anything after 'UGL'
        if "UGL" in clean_col:
            clean_col = clean_col.split("UGL")[0]
        # standardize LAT/LON
        if clean_col in ("LATITIDE", "LAT"):
            clean_col = "LATITUDE"
        elif clean_col in ("LON", "LONG"):
            clean_col = "LONGITUDE"

        if clean_col == "STATUS":
            if filename == "ENV_EPHT-cyanobacteria-season-summary-2013.xls":
                clean_col = "STATUS_DROP"
            else:
                clean_col = "BLOOMINTENSITY"

        new_columns.append(clean_col)

    df.columns = new_columns
    df["REPORTDATE"] = pd.to_datetime(df["REPORTDATE"], errors="coerce").dt.strftime(
        "%m/%d/%Y"
    )
    return df.drop(
        columns=[
            "CYLINDROSPERMOPSIN",  # Only used in 2015-2019
            "PLANKTONSAMPLEMETHOD",  # only used in 2016-2018 and 2020-2022
            "POTENTIALLYTOXICCYANOBACTERIACELLSML",  # only used in 2012
            "POTENTIALLYTOXICCYANOCELLSML",  # only used in 2014-2015
            "CYANOBACTERIADENSITYCELLSML",  # only used in 2016-2019
            "CLINDROSPERMOPSIN",  # only used in 2020
            "APPROXIMATEOFFSHORELENGTHBLOOM",  # only used in 2021-2022
            "APPROXIMATESHORELENGTHOFBLOOM",  # only used in 2021-2022
            "ACCESSTOSAMPLESITE",  # only used in 2021
            "METHOD",  # only used in 2013-2018 and 2020
            "MONITOREXPERIENCE",  # only used in 2014-2020
            "REPORTEREXPERIENCE",  # only used in 2013
            "OBJECTID",  # only used in 2013
            "BLOOMDISAPPEARED",  # only used in 2013
            "STATUSMOD",  # only used in 2013
            "WINDDIRECTION",  # only used in 2013
            "ALGAECOLOR",  # only used in 2013
            "BLOOMEXTENT",  # only used in 2013
            "DENSITY",  # only used in 2013, 2019, and 2022
            "COLLECTOR",  # only used in 2012-2013
            "ADDITIONALDETAILS",  # only used in 2019
            "ASSESSMENTMETHOD",  # only used in 2012-2019
            "DENSITYCELLSPERML",  # only used in 2021
            "CYN",  # only used in 2021-2022
            "SITEID",  # only used in 2012 as an abbreviation of STATION
            "REPORTTYPE",  # only used in 2019
            "SAMPLETYPE",  # only used in 2019
            "REPORTLOCATIONNAME",  # only used in 2012 and nearly identical to STATION
            "STATUS_DROP",  # only used in 2013
            "CYANOTAXACELLSML",
        ],
        errors="ignore",
    )


def _read_zipfile_files_into_combined_df(src: str, read_fn: Callable) -> pd.DataFrame:
    yearly_dfs: List[pd.DataFrame] = []
    with zipfile.ZipFile(src, "r") as archive:
        for filename in archive.namelist():
            with archive.open(filename) as report:
                # Skip directories and hidden files and files
                zipfilename = os.path.split(report.name)[-1]
                if not zipfilename or zipfilename.startswith("."):
                    continue
                try:
                    # Read the html document and extract the first element as there will only ever be 1 table.
                    df = read_fn(report)
                except UnicodeDecodeError as e:
                    print(f"Error reading file {filename}. Skipping...\n{e}")
                    continue
                yearly_dfs.append(df)
    return pd.concat(yearly_dfs)


def main():
    create_unified_csv_datasets()


if __name__ == "__main__":
    main()
