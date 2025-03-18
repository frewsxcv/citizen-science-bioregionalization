import geojson
import polars as pl
import shapely
import os
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

    for cluster, boundary, color in (
        cluster_boundary_dataframe.df
        .join(cluster_colors_dataframe.df, on="cluster")
        .iter_rows()
    ):
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
    """
    # Get the cluster boundary
    boundary_bytes = cluster_boundary_dataframe.get_boundary_for_cluster(cluster_id)
    cluster_boundary = shapely.from_wkb(boundary_bytes)
    
    # Load the ocean geojson
    ocean_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ocean.geojson')
    with open(ocean_file_path, 'r') as f:
        ocean_data = geojson.load(f)
    
    # Convert ocean geometries to shapely objects
    ocean_geometries = []
    for feature in ocean_data['features']:
        geom = shapely.geometry.shape(feature['geometry'])
        ocean_geometries.append(geom)
    
    if not ocean_geometries:
        return False
    
    # Create a single ocean polygon (union of all ocean geometries)
    ocean = shapely.unary_union(ocean_geometries)
    
    # Calculate the intersection area
    intersection = cluster_boundary.intersection(ocean)
    
    # Calculate the percentage of cluster area that is ocean
    cluster_area = cluster_boundary.area
    if cluster_area == 0:
        return False
    
    ocean_coverage = intersection.area / cluster_area
    
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
    """
    # Load the ocean geojson
    ocean_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ocean.geojson')
    with open(ocean_file_path, 'r') as f:
        ocean_data = geojson.load(f)
    
    # Convert ocean geometries to shapely objects
    ocean_geometries = []
    for feature in ocean_data['features']:
        geom = shapely.geometry.shape(feature['geometry'])
        ocean_geometries.append(geom)
    
    if not ocean_geometries:
        return []
    
    # Create a single ocean polygon (union of all ocean geometries)
    ocean = shapely.unary_union(ocean_geometries)
    
    ocean_clusters = []
    
    # Process each cluster
    for cluster_id, boundary_bytes in cluster_boundary_dataframe.df.select(["cluster", "boundary"]).iter_rows():
        cluster_boundary = shapely.from_wkb(boundary_bytes)
        
        # Calculate the intersection area
        intersection = cluster_boundary.intersection(ocean)
        
        # Calculate the percentage of cluster area that is ocean
        cluster_area = cluster_boundary.area
        if cluster_area == 0:
            continue
        
        ocean_coverage = intersection.area / cluster_area
        
        # Add to the list if the coverage exceeds the threshold
        if ocean_coverage >= threshold:
            ocean_clusters.append(cluster_id)
    
    return ocean_clusters
