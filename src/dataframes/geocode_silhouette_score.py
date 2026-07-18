
import polars as pl
import bioregion_rs
from src.matrices.geocode_distance import GeocodeDistanceMatrix

def build_geocode_silhouette_score_df(
    distance_matrix: GeocodeDistanceMatrix,
    geocode_cluster_df: pl.DataFrame,
) -> pl.DataFrame:
    """Build silhouette scores for all clustering results.

    Args:
        distance_matrix: Precomputed distance matrix between geocodes
        geocode_cluster_df: DataFrame with clustering results for all k values

    Returns:
        DataFrame with silhouette scores for all k values tested
    """
    df = bioregion_rs.build_geocode_silhouette_score(
        distance_matrix.condensed().tolist(), geocode_cluster_df
    )
    return df
