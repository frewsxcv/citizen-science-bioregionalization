# src/dataframes/permanova_results.py
import polars as pl
import logging


import bioregion_rs
from src.matrices.geocode_distance import GeocodeDistanceMatrix

logger = logging.getLogger(__name__)

def build_permanova_results_df(
    geocode_distance_matrix: GeocodeDistanceMatrix,
    geocode_cluster_df: pl.DataFrame,
    geocode_lf: pl.LazyFrame,
    permutations: int = 999,  # Default permutations
) -> pl.DataFrame:
    """
    Runs the PERMANOVA test and stores the results.

    Asserts that all geocodes in the distance matrix have cluster assignments.
    Raises exceptions if distance matrix or PERMANOVA calculation fails.

    Args:
        geocode_distance_matrix: Wrapper containing the condensed distance matrix.
        geocode_cluster_df: DataFrame mapping geocodes to clusters.
        geocode_lf: LazyFrame containing the ordered list of geocodes corresponding
                    to the distance matrix.
        permutations: Number of permutations for the test.

    Returns:
        A validated DataFrame conforming to PermanovaResultsSchema.
    """
    logger.info(
        f"build_permanova_results_df: Starting with {permutations} permutations"
    )

    geocode_ids = (
        geocode_lf.select("geocode").collect(engine="streaming")["geocode"].to_list()
    )

    df = bioregion_rs.build_permanova_results(
        geocode_distance_matrix.condensed().tolist(),
        geocode_ids,
        geocode_cluster_df.select("geocode", "cluster"),
        permutations,
    )

    return df
