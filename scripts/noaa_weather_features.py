import pandas as pd
import numpy as np

raw_data = "data/raw_files/NOAA_weather_data.csv"
output_csv = "data/unified_csvs/noaa_weather_avg.csv"

# Load CSV
df = pd.read_csv(raw_data)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 0)

df["DATE"] = pd.to_datetime(df["DATE"])
df = df.set_index("DATE")
df = df.loc["2012-01-01":"2024-12-31"] # Time range is 2012-2024

# More Descriptive Names
names_descriptive = {
"STATION" : "station",
"NAME" : "station_name",
"PRCP" : "precipitation",
"SNOW" : "snowfall",
"SNWD" : "snow_depth",
"PSUN" : "sunshine_percent",
"TSUN" : "sunshine_total",
"TAVG" : "air_temp_mean",
"TMAX" : "air_temp_max",
"TMIN" : "air_temp_min",
"WESD" : "water_on_ground",
"AWND" : "wind_speed_mean",
"FMTM" : "wind_fastest_time",
"PGTM" : "peak_gust_time",
"WDF2" : "wind_direction_2_min",
"WDF5" : "wind_direction_5_min",
"WSF2" : "wind_speed_2_min",
"WSF5" : "wind_speed_5_min",
"WT01" : "fog",
"WT02" : "heavy_fog",
"WT03" : "thunder",
"WT04" : "sleet_hail",
"WT05" : "hail",
"WT06" : "glaze_rime",
"WT07" : "dust_ash",
"WT08" : "smoke_haze",
"WT09" : "drifting_snow",
"WT11" : "high_winds",
"WT13" : "mist",
"WT14" : "drizzle",
"WT15" : "freezing_drizzle",
"WT16" : "rain",
"WT17" : "freezing_rain",
"WT18" : "snow_pellets",
"WT19" : "unknown_precipitation",
"WT21" : "ground_fog",
"WT22" : "ice_fog",
"WV01" : "fog_vicinity",
"WV03" : "thunder_vicinity"
}

# Rename columns
df = df.rename(columns=names_descriptive)

# Sort columns by non-null values
df_numeric = df.select_dtypes(include="number")

coverage = pd.DataFrame({
    "non_null_count": df_numeric.notna().sum(),
    "non_null_percentage": df_numeric.notna().mean()
}).sort_values("non_null_percentage", ascending=False)

# Drop features below threshold
THRESHOLD = 0.8 # Minimum data coverage to be a usable feature

usable_features = coverage.loc[coverage["non_null_percentage"] >= THRESHOLD].index.tolist()
df_numeric = df_numeric[usable_features]

# Trailing 14-day average computation
weekly_avg = df_numeric.shift(1).rolling(window=14, min_periods=1).mean()
weekly_avg = weekly_avg.rename(columns=lambda c: f"noaa_{c}")

# Export averages to CSV
weekly_avg.reset_index().to_csv(output_csv, index=False)