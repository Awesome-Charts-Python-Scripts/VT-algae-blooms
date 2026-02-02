"""Combine VT Department of Environmental Conservation reports from multiple years into a single CSV.

Report files can be obtained from https://anrweb.vermont.gov/DEC/_DEC/LongTermMonitoringLakes.aspx, selecting
a year under the "Data by Year" dropdown, and clicking "View Selected Data". Reports are downloaded as xls files
but their content is actually html. Report files must be zipped into a single folder in order to execute this
program.

Usage:
    python scripts/combine_dec_yearly_reports.py \
        -i data/raw_files/vt_dec_yearly_reports.zip \
        -o data/unified_csvs/vt_dec.csv
"""

import os
import zipfile
import argparse
import pandas as pd
from typing import List


def read_and_combine_dec_files_into_unified_csv(src: str, dst: str):
    yearly_dfs: List[pd.DataFrame] = []
    with zipfile.ZipFile(src, "r") as archive:
        for filename in archive.namelist():
            with archive.open(filename) as report:
                try:
                    # Read the html document and extract the first element as there will only ever be 1 table.
                    df = pd.read_html(report)[0]
                except UnicodeDecodeError:
                    print(f"Error reading file {filename}. Skipping...")
                yearly_dfs.append(df)

    pd.concat(yearly_dfs).to_csv(dst, index=False)
    os.chmod(
        dst, 0o777
    )  # Open up all the file permissions (read/write/execute for all)


def main():
    parser = argparse.ArgumentParser(
        description="Combine VT Department of Environmental Conservation xls reports from multiple years into a single CSV"
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
    read_and_combine_dec_files_into_unified_csv(src=args.i, dst=args.o)


if __name__ == "__main__":
    main()
