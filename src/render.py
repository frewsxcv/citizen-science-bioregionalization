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

    # Create plot with polars-st
    df_st: polars_st.GeoDataFrameNameSpace = (
        df.select(
            pl.col("geometry"),
            pl.col("cluster"),
            pl.col("fillColor"),
            pl.col("color"),
        )
        .st  # type: ignore
    )
    plot = (
        df_st.plot(
            color="fillColor",
            fillOpacity=0.5,
            strokeWidth=1.0,
            stroke="color",
        )
        .project(type="identity")
        .encode(fill="properties.fillColor:N")
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

    # Create plot
    df_st: polars_st.GeoDataFrameNameSpace = (
        df.select(
            pl.col("geometry"),
            pl.col("fillColor"),
            pl.col("color"),
        )
        .st  # type: ignore
    )
    plot = df_st.plot(
        color="fillColor",
        fillOpacity=0.5,
        strokeWidth=1.0,
        stroke="color",
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

    # Create plot
    df_st: polars_st.GeoDataFrameNameSpace = (
        df.select(
            pl.col("geometry"),
            pl.col("cluster"),
            pl.col("fillColor"),
            pl.col("color"),
        )
        .st  # type: ignore
    )
    plot = (
        df_st.plot(
            color="fillColor",
            fillOpacity=0.7,
            strokeWidth=0.8,
            stroke="color",
        )
        .encode(fill="properties.fillColor:N")
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
