import geojson
import polars as pl
import polars_st
import io
import base64
import shapely
from typing import Optional, List, Dict, Any, cast, Union, BinaryIO, TextIO
from src import output


def plot_clusters(
    feature_collection: geojson.FeatureCollection,
    file_obj: Optional[Union[TextIO, BinaryIO]] = None,
) -> None:
    """
    Plot all clusters using polars-st spatial plotting.

    Args:
        feature_collection: GeoJSON feature collection containing all clusters
        file_obj: File-like object to save the image to (optional)
    """
    df = features_to_polars_df(feature_collection["features"])

    # Darken fill colors for stroke
    darkened_colors = darken_hex_colors_polars(df["fill"])

    # Create plot with polars-st
    df_st: polars_st.GeoDataFrameNameSpace = (
        df.with_columns(darkened_fill=darkened_colors)
        .select(
            pl.col("geometry"),
            pl.col("cluster"),
            pl.col("fill"),
            pl.col("darkened_fill"),
        )
        .st  # type: ignore
    )
    plot = (
        df_st.plot(
            color="fill",
            fillOpacity=0.5,
            strokeWidth=1.0,
            stroke="darkened_fill",
        )
        .project(type="identity")
        .encode(fill="properties.fill:N")
    )

    if file_obj:
        # Save to the file-like object - specify format as png
        plot.save(file_obj, format="png")
    else:
        # Display the plot if not saving
        plot.show()


def plot_single_cluster(
    feature_collection: geojson.FeatureCollection,
    cluster_id: int,
    file_obj: Optional[Union[TextIO, BinaryIO]] = None,
) -> None:
    """
    Plot a single cluster and optionally save it to a file.

    Args:
        feature_collection: GeoJSON feature collection containing all clusters
        cluster_id: The ID of the cluster to plot
        file_obj: File-like object to save the image to (optional)
    """
    # Filter features for the specific cluster
    cluster_features = [
        feature
        for feature in feature_collection["features"]
        if feature["properties"]["cluster"] == cluster_id
    ]

    # Convert to polars DataFrame
    df = features_to_polars_df(cluster_features)

    # Darken fill colors for stroke
    darkened_colors = darken_hex_colors_polars(df["fill"])

    # Create plot
    df_st: polars_st.GeoDataFrameNameSpace = (
        df.with_columns(darkened_fill=darkened_colors)
        .select(
            pl.col("geometry"),
            pl.col("fill"),
            pl.col("darkened_fill"),
        )
        .st  # type: ignore
    )
    plot = df_st.plot(
        color="fill",
        fillOpacity=0.5,
        strokeWidth=1.0,
        stroke="darkened_fill",
    ).project(type="identity", reflectY=True)

    # Handle output
    if file_obj:
        # Save to the file-like object - specify format as png
        plot.save(file_obj, format="png")
    else:
        plot.show()


def plot_entire_region(
    feature_collection: geojson.FeatureCollection,
    file_obj: Optional[Union[TextIO, BinaryIO]] = None,
) -> None:
    """
    Plot the entire region with all clusters.

    Args:
        feature_collection: GeoJSON feature collection containing all clusters
        file_obj: File-like object to save the image to (optional)
    """
    # Convert to polars DataFrame
    df = features_to_polars_df(feature_collection["features"])

    # Darken fill colors for stroke with a factor of 0.3
    darkened_colors = darken_hex_colors_polars(df["fill"], factor=0.3)

    # Create plot
    df_st: polars_st.GeoDataFrameNameSpace = (
        df.with_columns(darkened_fill=darkened_colors)
        .select(
            pl.col("geometry"),
            pl.col("cluster"),
            pl.col("fill"),
            pl.col("darkened_fill"),
        )
        .st  # type: ignore
    )
    plot = (
        df_st.plot(
            color="fill",
            fillOpacity=0.7,
            strokeWidth=0.8,
            stroke="darkened_fill",
        )
        .encode(fill="properties.fill:N")
        .project(type="identity", reflectY=True)
    )

    # Handle output
    if file_obj:
        # Save to the file-like object - specify format as png
        plot.save(file_obj, format="png")
    else:
        plot.show()


def features_to_polars_df(features: List[Dict[str, Any]]) -> pl.DataFrame:
    """
    Convert GeoJSON features to a polars DataFrame with geometry column.

    Args:
        features: List of GeoJSON features

    Returns:
        Polars DataFrame with geometry and properties columns
    """
    rows = []
    for feature in features:
        # Convert GeoJSON geometry to shapely geometry and then to WKB binary format
        shapely_geom = shapely.geometry.shape(feature["geometry"])
        row = {"geometry": shapely.to_wkb(shapely_geom)}

        # Add all the properties
        row.update(feature["properties"])
        rows.append(row)

    return pl.DataFrame(rows)


def darken_hex_color(hex_color: str, factor: float = 0.5) -> str:
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


def darken_hex_colors_polars(hex_colors: pl.Series, factor: float = 0.5) -> pl.Series:
    """
    Darkens each hex color in a polars Series by the given factor.

    Args:
        hex_colors: A polars Series of hex color strings
        factor: A float between 0 and 1 (0 = black, 1 = original color)

    Returns:
        A polars Series of darkened hex colors
    """
    return pl.Series(
        [darken_hex_color(color, factor) for color in hex_colors.to_list()]
    )
