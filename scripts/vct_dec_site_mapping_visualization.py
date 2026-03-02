"""Create a plot of Lake Champlain with the VCT and DEC monitoring sites overlaid on top.

Shapefile data was obtained from: https://geodata.vermont.gov/datasets/vt-lake-champlain-extracted-from-vhdcarto-polygon/about

The DEC monitoring sites do not map 1-to-1 with our target sites in the VCT dataset. By plotting
on a map, we can visualize all site locations and determine the best mappings between them. An alternative
approach would be to select the nearest locations using euclidean distance with some distance cutoff, however,
because the number of sites to map is small (less than 20), we opted to manually map sites for simplicity.

Using this script, the following mapping was manually determined which was saved to VCT_to_DEC_site_mappings.json
- Missisquoi Bay -> Missisquoi Bay, Missisquoi Bay Central
- St. Albans Bay -> St. Albans Bay
- Inland Sea -> None, possibly northeast arm
- Malletts Bay -> Malletts Bay
- Main Lake Central -> Main Lake, Burlington Bay
- Main Lake North -> None, possibly Otter Creek Segment
- Main Lake South -> Port Henry Segment

Usage:
    python scripts/vct_dec_site_mapping_visualization.py
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

LAKE_CHAMPLAIN_SHAPEFILE_PATH = (
    "data/shapefiles/VT_Lake_Champlain_(extracted_from_VHDCARTO).shp"
)
TARGET_CSV_PATH = "data/unified_csvs/vct_unified_prepped.csv"
DEC_STATION_LOCATIONS_CSV = "data/data_dictionaries/DEC_station_locations.csv"


def display_site_locations_map():
    # Read the Lake Champlain shapefile into a geodataframe and project to WGS84 coordinate reference system
    # so that coordinates are referenced in lat/lon format.
    lake_champlain_gdf = gpd.read_file(LAKE_CHAMPLAIN_SHAPEFILE_PATH).to_crs(epsg=4326)

    # Read in the VCT and DEC site locations
    vct_sites = pd.read_csv(TARGET_CSV_PATH)[
        ["region", "latitude", "longitude"]
    ].drop_duplicates()
    dec_sites = pd.read_csv(DEC_STATION_LOCATIONS_CSV)[
        ["DEC_station", "degrees_latitude", "degrees_longitude"]
    ].drop_duplicates()

    # Load the VCT and DEC lat/lon locations as a geodataframe
    vct_points_gdf = gpd.GeoDataFrame(
        vct_sites,
        geometry=gpd.points_from_xy(vct_sites["longitude"], vct_sites["latitude"]),
        crs="EPSG:4326",
    )
    dec_points_gdf = gpd.GeoDataFrame(
        dec_sites,
        geometry=gpd.points_from_xy(
            dec_sites["degrees_longitude"], dec_sites["degrees_latitude"]
        ),
        crs="EPSG:4326",
    )

    # Plot the shapefile and site locations
    fig, ax = plt.subplots(figsize=(8, 8))
    lake_champlain_gdf.plot(ax=ax, color="lightgrey", edgecolor="black")
    vct_points_gdf.plot(
        ax=ax, marker="o", color="red", markersize=10, label="VCT Sites"
    )
    dec_points_gdf.plot(
        ax=ax, marker="o", color="blue", markersize=10, label="DEC Sites"
    )
    plt.legend(loc="upper left", bbox_to_anchor=(1.0, 1.05))
    # Add the site names to the plot at each point location
    for _, row in vct_sites.iterrows():
        ax.annotate(
            row["region"],
            xy=(row["longitude"], row["latitude"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    for _, row in dec_sites.iterrows():
        ax.annotate(
            row["DEC_station"],
            xy=(row["degrees_longitude"], row["degrees_latitude"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    plt.show()


def main():
    display_site_locations_map()


if __name__ == "__main__":
    main()
