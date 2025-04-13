import geojson
import polars as pl
import shapely
import os
from shapely.geometry import shape, Polygon, MultiPolygon
from shapely.ops import unary_union
from typing import Union

from src.dataframes.cluster_boundary import ClusterBoundaryDataFrame
from src.types import Geocode, ClusterId
from src.dataframes.cluster_color import ClusterColorDataFrame
from src.dataframes.geocode_cluster import GeocodeClusterDataFrame
from src import output


def build_geojson_feature(
    geometry: shapely.Geometry,
    cluster: ClusterId,
    color: str,
) -> geojson.Feature:
    return geojson.Feature(
        properties={
            # "label": ", ".join(geocodes),
            "fill": color,
            "stroke-width": 0,
            "cluster": cluster,
        },
        geometry=shapely.geometry.mapping(geometry),  # type: ignore
    )


def build_geojson_feature_collection(
    cluster_boundary_dataframe: ClusterBoundaryDataFrame,
    cluster_colors_dataframe: ClusterColorDataFrame,
) -> geojson.FeatureCollection:
    features: list[geojson.Feature] = []

    for cluster, boundary, color in cluster_boundary_dataframe.df.join(
        cluster_colors_dataframe.df, on="cluster"
    ).iter_rows():
        features.append(
            build_geojson_feature(
                shapely.from_wkb(boundary),
                cluster,
                color,
            )
        )
    return geojson.FeatureCollection(features)


def write_geojson(
    feature_collection: geojson.FeatureCollection, output_file: str
) -> None:
    # Prepare the output file path
    output_file = output.prepare_file_path(output_file)

    with open(output_file, "w") as geojson_writer:
        geojson.dump(feature_collection, geojson_writer)


def _load_ocean_geometry() -> Union[Polygon, MultiPolygon]:
    """
    Loads the ocean geojson file and returns a unified Shapely geometry.

    Raises:
        FileNotFoundError: If the ocean.geojson file cannot be found.
        ValueError: If the ocean.geojson file is empty or contains no valid features.
    """
    ocean_file_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ocean.geojson"
    )
    # Let FileNotFoundError propagate if the file doesn't exist
    with open(ocean_file_path, "r") as f:
        ocean_data = geojson.load(f)

    ocean_geometries = [
        shape(feature["geometry"]) for feature in ocean_data.get("features", [])
    ]

    if not ocean_geometries:
        raise ValueError(f"No valid geometries found in {ocean_file_path}")

    # Create a single ocean polygon (union of all ocean geometries)
    return unary_union(ocean_geometries)


def _calculate_ocean_coverage(
    cluster_boundary: shapely.Geometry, ocean_geometry: Union[Polygon, MultiPolygon]
) -> float:
    """Calculates the fraction of the cluster boundary area covered by the ocean."""
    if not cluster_boundary or not ocean_geometry:
        return 0.0

    # Calculate the intersection area
    intersection = cluster_boundary.intersection(ocean_geometry)

    # Calculate the percentage of cluster area that is ocean
    cluster_area = cluster_boundary.area
    if cluster_area == 0:
        return 0.0  # Avoid division by zero

    return intersection.area / cluster_area


def is_cluster_mostly_ocean(
    cluster_boundary_dataframe: ClusterBoundaryDataFrame,
    cluster_id: ClusterId,
    threshold: float = 0.90,
) -> bool:
    """
    Determines if a cluster's boundary is almost entirely within the ocean.

    Args:
        cluster_boundary_dataframe: DataFrame containing cluster boundaries
        cluster_id: ID of the cluster to check
        threshold: Fraction of area required to be within ocean (default: 0.90 or 90%)

    Returns:
        bool: True if cluster is mostly in ocean, False otherwise

    Raises:
        FileNotFoundError: If the ocean.geojson file cannot be found.
        ValueError: If the ocean.geojson file is empty or contains no valid features,
                    or if the provided cluster_id is not found in the dataframe.
    """
    # _load_ocean_geometry will raise FileNotFoundError or ValueError if it fails
    ocean_geometry = _load_ocean_geometry()

    # Get the cluster boundary
    boundary_bytes = cluster_boundary_dataframe.get_boundary_for_cluster(cluster_id)
    if boundary_bytes is None:
        raise ValueError(f"Cluster ID {cluster_id} not found in dataframe.")

    cluster_boundary = shapely.from_wkb(boundary_bytes)

    # Calculate the ocean coverage
    ocean_coverage = _calculate_ocean_coverage(cluster_boundary, ocean_geometry)

    # Return True if the coverage exceeds the threshold
    return ocean_coverage >= threshold


def find_ocean_clusters(
    cluster_boundary_dataframe: ClusterBoundaryDataFrame,
    threshold: float = 0.90,
) -> list[ClusterId]:
    """
    Finds all clusters that are almost entirely within the ocean.

    This is more efficient than calling is_cluster_mostly_ocean() for each cluster
    since it loads the ocean data only once.

    Args:
        cluster_boundary_dataframe: DataFrame containing cluster boundaries
        threshold: Fraction of area required to be within ocean (default: 0.90 or 90%)

    Returns:
        list[ClusterId]: List of cluster IDs that are mostly in ocean

    Raises:
        FileNotFoundError: If the ocean.geojson file cannot be found.
        ValueError: If the ocean.geojson file is empty or contains no valid features.
    """
    # _load_ocean_geometry will raise FileNotFoundError or ValueError if it fails
    ocean_geometry = _load_ocean_geometry()

    ocean_clusters = []

    # Process each cluster
    for cluster_id, boundary_bytes in cluster_boundary_dataframe.df.select(
        ["cluster", "geometry"]
    ).iter_rows():
        cluster_boundary = shapely.from_wkb(boundary_bytes)

        # Calculate the ocean coverage
        ocean_coverage = _calculate_ocean_coverage(cluster_boundary, ocean_geometry)

        # Add to the list if the coverage exceeds the threshold
        if ocean_coverage >= threshold:
            ocean_clusters.append(cluster_id)

    return ocean_clusters
