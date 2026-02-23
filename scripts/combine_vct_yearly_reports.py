"""Combine VT Department of Environmental Conservation reports from multiple years into a single CSV.

Report files can be obtained from https://www.healthvermont.gov/environment/tracking/cyanobacteria-blue-green-algae-tracker,
scrolling down on the page below the map, and selecting a year under the "Season Summaries" dropdown. Reports are
downloaded as xlsx files. Report files must be zipped into a single folder in order to execute this program.

Usage:
    python scripts/combine_vct_yearly_reports.py \
        -i data/raw_files/vct_yearly_reports.zip \
        -o data/unified_csvs/vct.csv
"""

import os
import zipfile
import argparse
import pandas as pd
from typing import List


def read_and_combine_vct_files_into_unified_csv(src: str, dst: str):
    yearly_dfs: List[pd.DataFrame] = []
    with zipfile.ZipFile(src, "r") as archive:
        for filename in archive.namelist():
            with archive.open(filename) as report:
                try:
                    # Read the html document and extract the first element as there will only ever be 1 table.
                    df = pd.read_html(report)[0]
                except UnicodevctodeError:
                    print(f"Error reading file {filename}. Skipping...")
                yearly_dfs.append(df)

    pd.concat(yearly_dfs).to_csv(dst, index=False)
    os.chmod(
        dst, 0o777
    )  # Open up all the file permissions (read/write/execute for all)


def main():
    parser = argparse.ArgumentParser(
        description="Combine Vermont Cyanobacteria Tracker season summaries from multiple years into a single CSV"
    )
    # Optional argument (flag)
    parser.add_argument(
        "-i",
        action="store",
        required=True,
        help="Input zipfile containing xls files to combine",
    )
    parser.add_argument(
        "-o", action="store", default="out.csv", help="Output file name"
    )

    args = parser.parse_args()
    read_and_combine_vct_files_into_unified_csv(src=args.i, dst=args.o)


if __name__ == "__main__":
    main()
