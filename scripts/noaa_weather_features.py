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

# Weather Key Descriptions
descriptions = {
"STATION" : "Station identification code",
"NAME" : "Name of the station (city/airport name)",
"PRCP" : "Precipitation (mm)",
"SNOW" : "Snowfall (mm)",
"SNWD" : "Snow depth (mm)",
"PSUN" : "Daily percent of possible sunshine (percent)",
"TSUN" : "Daily total sunshine (minutes)",
"TAVG" : "Average temperature (Celcius)",
"TMAX" : "Maximum temperature (Celcius)",
"TMIN" : "Minimum temperature  (Celcius)",
"WESD" : "Water equivalent of snow on the ground (mm)",
"AWND" : "Average daily wind speed (meters per second)",
"FMTM" : "Time of fastest mile or 1-minute wind (hours and minutes HHMM)",
"PGTM" : "Peak gust time (hours and minutes HHMM)",
"WDF2" : "Direction of fastest 2-minute wind (degrees)",
"WDF5" : "Direction of fastest 5-minute wind (degrees)",
"WSF2" : "Fastest 2-minute wind speed (meters per second)",
"WSF5" : "Fastest 5-minute wind speed (meters per second)",
"WT01" : "Fog, ice fog, or freezing fog",
"WT02" : "Heavy fog or heavy freezing fog",
"WT03" : "Thunder",
"WT04" : "Ice pellets, sleet, snow pellets, or small hail",
"WT05" : "Hail",
"WT06" : "Glaze or rime",
"WT07" : "Dust, volcanic ash, blowing dust, blowing sand, or blowing obstruction",
"WT08" : "Smoke or haze",
"WT09" : "Blowing or drifting snow",
"WT11" : "High or damaging winds",
"WT13" : "Mist",
"WT14" : "Drizzle",
"WT15" : "Freezing drizzle",
"WT16" : "Rain",
"WT17" : "Freezing rain",
"WT18" : "Snow, snow pellets, snow grains, or ice crystals",
"WT19" : "Unknown source of precipitation",
"WT21" : "Ground fog",
"WT22" : "Ice fog or freezing fog",
"WV01" : "Fog, ice fog, or freezing fog",
"WV03" : "Thunder"
}

categories = {
"STATION" : "Metadata",
"NAME" : "Metadata",
"PRCP" : "Precipitation",
"SNOW" : "Precipitation",
"SNWD" : "Precipitation",
"PSUN" : "Sunshine",
"TSUN" : "Sunshine",
"TAVG" : "Air Temperature",
"TMAX" : "Air Temperature",
"TMIN" : "Air Temperature",
"WESD" : "Water",
"AWND" : "Wind",
"FMTM" : "Wind",
"PGTM" : "Wind",
"WDF2" : "Wind",
"WDF5" : "Wind",
"WSF2" : "Wind",
"WSF5" : "Wind",
"WT01" : "Weather",
"WT02" : "Weather",
"WT03" : "Weather",
"WT04" : "Weather",
"WT05" : "Weather",
"WT06" : "Weather",
"WT07" : "Weather",
"WT08" : "Weather",
"WT09" : "Weather",
"WT11" : "Weather",
"WT13" : "Weather",
"WT14" : "Weather",
"WT15" : "Weather",
"WT16" : "Weather",
"WT17" : "Weather",
"WT18" : "Weather",
"WT19" : "Weather",
"WT21" : "Weather",
"WT22" : "Weather",
"WV01" : "Weather in Vicinity",
"WV03" : "Weather in Vicinity"
}

# Sort columns by non-null values
df_numeric = df.select_dtypes(include="number")

coverage = pd.DataFrame({
    "non_null_count": df_numeric.notna().sum(),
    "non_null_percentage": df_numeric.notna().mean()
}).sort_values("non_null_percentage", ascending=False)

coverage["category"] = categories
coverage["description"] = descriptions

# Drop features below threshold
THRESHOLD = 0.8 # Minimum data coverage to be a usable feature

usable_features = coverage.loc[coverage["non_null_percentage"] >= THRESHOLD].index.tolist()
df_numeric = df_numeric[usable_features]

# Trailing 14-day average computation
weekly_avg = df_numeric.shift(1).rolling(window=14, min_periods=1).mean()
weekly_avg = weekly_avg.rename(columns=lambda c: f"noaa_{c}")

# Export averages to CSV
weekly_avg.reset_index().to_csv(output_csv, index=False)