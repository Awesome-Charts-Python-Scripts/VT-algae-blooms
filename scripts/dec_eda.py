"""Visualize VT DEC feature data to understand data trends and availability across monitoring sites.

Usage:
    python scripts/dec_eda.py \
        --threshold=0.3
"""

import argparse
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from create_dec_features import create_dec_features

FEATURE_COLUMNS_OF_INTEREST = [
    "total nitrogen",
    "total phosphorus",
    "dissolved phosphorus",
    "dissolved inorganic carbon",
    "dissolved organic carbon",
    "non-purgeable organic carbon",
    "dissolved oxygen",
    "chlorophyll-a",
    "secchi depth",
    "temperature",
    "net phytoplankton, chlorophyta biovolume",
    "net phytoplankton, chlorophyta density",
    "net phytoplankton, chrysophyta biovolume",
    "net phytoplankton, chrysophyta density",
    "net phytoplankton, cyanobacteria biovolume",
    "net phytoplankton, cyanobacteria density",
    "net phytoplankton, pyrrophyta biovolume",
    "net phytoplankton, pyrrophyta density",
    "net phytoplankton, total biovolume",
    "net phytoplankton, total density",
]


def visualize(availability_threshold: float = 0.0):
    """Visualize the availability of feature data after aggregating to the target observation dates.

    Args:
        availability_threshold: Only display features having at least one site meeting this availability threshold.
            There are a large number of features and the plot can be quite noisy. This helps to filter the features
            down to only those likely to be used.
    """
    feature_df = create_dec_features(FEATURE_COLUMNS_OF_INTEREST)

    # Get the list of columns meeting the availability threshold
    feature_df_availability = feature_df.groupby("region").apply(
        lambda grp: 1 - grp.isnull().mean()
    )
    columns_meeting_availability_threshold = list(
        set(
            feature_df_availability.max()[
                (feature_df_availability > availability_threshold).all()
            ].index
        )
        - {"report_date"}
    )

    # Generate a grid of plots for each station X feature
    fig, axs = plt.subplots(
        nrows=feature_df["region"].nunique(),
        ncols=len(columns_meeting_availability_threshold),
        figsize=(16, 8),
        layout="constrained",
    )
    cmap = ListedColormap(["green", "red"])  # green: present, red: missing

    # Iterate through all feature regions (target monitoring sites)
    plt_row = 0
    for target_site, grp in feature_df.groupby("region"):
        plt_col = 0
        for col in grp[columns_meeting_availability_threshold]:
            # Generate a colormap plot of all the missing (nan) data entries
            missing = grp[col].isna().astype(int)
            axs[plt_row, plt_col].imshow(
                [missing], aspect="auto", cmap=cmap, interpolation="none"
            )
            # Set the axis title which includes the percentage of data availability
            plot_title = f"{(1 - (grp[col].isnull().sum() / len(grp))) * 100:.2f}%"
            if plt_row == 0:
                plot_title = f"{'\n'.join(col.split(' '))}\n{plot_title}"
            axs[plt_row, plt_col].set_title(plot_title)
            # Remove axes ticks for better readability
            axs[plt_row, plt_col].set_xticks([])
            axs[plt_row, plt_col].set_yticks([])
            plt_col += 1
        plt_row += 1

    # Set the y-axis labels using the region names
    for ax, region in zip(axs[:, 0], feature_df["region"].unique()):
        ax.set_ylabel(
            "\n".join(region.split(" ")), rotation=0, size="large", labelpad=30
        )
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Generate a CSV of VT DEC features to correspond 1-to-1 with our targets."
    )
    parser.add_argument(
        "-t", "--threshold", action="store", default=0.0,
        help="Only plot features that have at least one site meeting this data availability threshold"
    )
    args = parser.parse_args()
    visualize(float(args.threshold))


if __name__ == "__main__":
    main()
