import dataframely as dy

import bioregion_rs
from src.dataframes.geocode_cluster import GeocodeClusterMultiKSchema
from src.matrices.geocode_distance import GeocodeDistanceMatrix


class GeocodeSilhouetteScoreSchema(dy.Schema):
    geocode = dy.UInt64(nullable=True)
    silhouette_score = dy.Float64(nullable=False)
    num_clusters = dy.UInt32(nullable=False)


def build_geocode_silhouette_score_df(
    distance_matrix: GeocodeDistanceMatrix,
    geocode_cluster_df: dy.DataFrame[GeocodeClusterMultiKSchema],
) -> dy.DataFrame[GeocodeSilhouetteScoreSchema]:
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
    return GeocodeSilhouetteScoreSchema.validate(df)
