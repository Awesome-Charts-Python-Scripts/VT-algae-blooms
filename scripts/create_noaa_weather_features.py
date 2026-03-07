import pandas as pd
import numpy as np

raw_data = "data/raw_files/NOAA_weather_data.csv"
output_csv = "data/unified_csvs/noaa_weather_avg.csv"


df = pd.read_csv(raw_data)

# pd.set_option("display.max_columns", None)
# pd.set_option("display.width", 0)

df["DATE"] = pd.to_datetime(df["DATE"])
# df["STATION"] = df["STATION"].astype(str)
# df["NAME"] = df["NAME"].astype(str)
# df["DATE"] = df["DATE"].
df = df.set_index("DATE")
# df = df[(df["DATE"] >= "2012-01-01") & (df["DATE"] <= "2024-12-31")]
df = df.loc["2012-01-01":"2024-12-31"] # Time range is 2012-2024
