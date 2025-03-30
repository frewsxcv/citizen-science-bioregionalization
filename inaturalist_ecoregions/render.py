import geopandas
import matplotlib.pyplot as plt
import contextily
import geojson
from matplotlib.lines import Line2D
import polars as pl
import io
import base64
import os
from typing import Optional
from inaturalist_ecoregions import output


def plot_clusters(
    feature_collection: geojson.FeatureCollection, save_path: Optional[str] = None
) -> None:
    geojson_gdf = geopandas.GeoDataFrame.from_features(
        feature_collection["features"], crs="EPSG:4326"
    )
    geojson_gdf_wm = geojson_gdf.to_crs(epsg=3857)

    ax = geojson_gdf_wm.plot(
        color=geojson_gdf_wm["fill"],
        categorical=True,
        linewidth=0,
        alpha=0.5,
    )
    geojson_gdf_wm.boundary.plot(
        ax=ax,
        color=darken_hex_colors_series(geojson_gdf_wm["fill"], factor=0.5),
        linewidth=1.0,
        alpha=1,
    )

    # Add a legend
    cluster_and_fill = geojson_gdf_wm[["cluster", "fill"]]
    custom_points = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=10, color=row.fill)
        for row in cluster_and_fill.itertuples()
    ]
    leg_points = ax.legend(custom_points, cluster_and_fill["cluster"].unique())
    ax.add_artist(leg_points)

    contextily.add_basemap(
        ax, source=contextily.providers.CartoDB.Positron, attribution_size=0
    )
    ax.set_axis_off()

    if save_path:
        # Prepare the output file path
        save_path = output.prepare_file_path(save_path)
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


def plot_single_cluster(
    feature_collection: geojson.FeatureCollection,
    cluster_id: int,
    save_path: Optional[str] = None,
    to_base64: bool = False,
) -> str:
    """
    Plot a single cluster and optionally save it to a file or return as base64 encoded image.

    Args:
        feature_collection: GeoJSON feature collection containing all clusters
        cluster_id: The ID of the cluster to plot
        save_path: Path to save the image to (optional)
        to_base64: Whether to return the image as a base64 encoded string

    Returns:
        Base64 encoded image string if to_base64 is True, otherwise empty string
    """
    # Filter features for the specific cluster
    cluster_features = [
        feature
        for feature in feature_collection["features"]
        if feature["properties"]["cluster"] == cluster_id
    ]

    # Create a new feature collection with just this cluster
    cluster_feature_collection = geojson.FeatureCollection(cluster_features)

    # Convert to GeoDataFrame
    geojson_gdf = geopandas.GeoDataFrame.from_features(
        cluster_features, crs="EPSG:4326"
    )
    geojson_gdf_wm = geojson_gdf.to_crs(epsg=3857)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the cluster
    geojson_gdf_wm.plot(
        ax=ax,
        color=geojson_gdf_wm["fill"],
        categorical=True,
        linewidth=0,
        alpha=0.5,
    )
    geojson_gdf_wm.boundary.plot(
        ax=ax,
        color=darken_hex_colors_series(geojson_gdf_wm["fill"], factor=0.5),
        linewidth=1.0,
        alpha=1,
    )

    # Add basemap
    contextily.add_basemap(
        ax, source=contextily.providers.CartoDB.Positron, attribution_size=0
    )

    # Set title and remove axes
    ax.set_title(f"Cluster {cluster_id}")
    ax.set_axis_off()

    # Tight layout
    plt.tight_layout()

    # Handle output
    if save_path:
        save_path = output.prepare_file_path(save_path)
        plt.savefig(save_path, bbox_inches="tight")

    if to_base64:
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode("ascii")
        plt.close(fig)
        return img_str

    plt.close(fig)
    return ""


def plot_entire_region(
    feature_collection: geojson.FeatureCollection,
    to_base64: bool = False,
) -> str:
    """
    Plot the entire region with all clusters and return as base64 encoded image.

    Args:
        feature_collection: GeoJSON feature collection containing all clusters
        to_base64: Whether to return the image as a base64 encoded string

    Returns:
        Base64 encoded image string if to_base64 is True, otherwise empty string
    """
    # Convert to GeoDataFrame
    geojson_gdf = geopandas.GeoDataFrame.from_features(
        feature_collection["features"], crs="EPSG:4326"
    )
    geojson_gdf_wm = geojson_gdf.to_crs(epsg=3857)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot all clusters
    geojson_gdf_wm.plot(
        ax=ax,
        color=geojson_gdf_wm["fill"],
        categorical=True,
        linewidth=0,
        alpha=0.7,
    )
    geojson_gdf_wm.boundary.plot(
        ax=ax,
        color=darken_hex_colors_series(geojson_gdf_wm["fill"], factor=0.3),
        linewidth=0.8,
        alpha=1,
    )

    # Add a legend
    cluster_and_fill = geojson_gdf_wm[["cluster", "fill"]].drop_duplicates()
    legend_items = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=10, color=row.fill)
        for row in cluster_and_fill.itertuples()
    ]
    cluster_ids = sorted(cluster_and_fill["cluster"].unique())
    leg_points = ax.legend(
        legend_items,
        [f"Cluster {c}" for c in cluster_ids],
        title="Clusters",
        loc="best",
    )
    ax.add_artist(leg_points)

    # Add basemap
    contextily.add_basemap(
        ax, source=contextily.providers.CartoDB.Positron, attribution_size=8
    )

    # Set title and remove axes
    ax.set_title("All Ecoregion Clusters")
    ax.set_axis_off()

    # Tight layout
    plt.tight_layout()

    # Handle output
    if to_base64:
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode("ascii")
        plt.close(fig)
        return img_str

    plt.close(fig)
    return ""


def darken_hex_color(hex_color, factor=0.5):
    """
    Darkens a hex color by multiplying RGB components by the given factor.

    Args:
        hex_color: A hex color string like '#ff0000' or '#f00'
        factor: A float between 0 and 1 (0 = black, 1 = original color)

    Returns:
        A darkened hex color string
    """
    # Remove the # if present
    hex_color = hex_color.lstrip("#")

    # Handle shorthand hex format (#rgb)
    if len(hex_color) == 3:
        hex_color = "".join([c * 2 for c in hex_color])

    # Convert hex to RGB
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    # Darken each component
    r = int(r * factor)
    g = int(g * factor)
    b = int(b * factor)

    # Convert back to hex
    return f"#{r:02x}{g:02x}{b:02x}"


def darken_hex_colors_series(hex_colors_series: geopandas.GeoSeries, factor=0.5):
    """
    Darkens each hex color in a pandas Series by the given factor.

    Args:
        hex_colors_series: A pandas Series of hex color strings
        factor: A float between 0 and 1 (0 = black, 1 = original color)

    Returns:
        A pandas Series of darkened hex colors
    """
    return hex_colors_series.apply(lambda x: darken_hex_color(x, factor))
