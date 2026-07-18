import polars as pl
import logging


import bioregion_rs

logger = logging.getLogger(__name__)

def build_cluster_boundary_df(
    geocode_cluster_df: pl.DataFrame,
    geocode_lf: pl.LazyFrame,
) -> pl.DataFrame:
    """Build cluster boundaries by combining geocode boundaries.

    Creates a single boundary polygon for each cluster by unioning all
    the hexagon boundaries belonging to that cluster.

    Args:
        geocode_cluster_df: DataFrame mapping geocodes to clusters
        geocode_lf: LazyFrame containing geocode boundary geometries

    Returns:
        A validated DataFrame conforming to ClusterBoundarySchema
    """
    logger.info("build_cluster_boundary_df: Starting")

    geocode_df = geocode_lf.select("geocode", "boundary").collect(engine="streaming")

    df = bioregion_rs.build_cluster_boundary(geocode_cluster_df, geocode_df)

    logger.info(f"build_cluster_boundary_df: Output has {df.height} cluster boundaries")

    return df
