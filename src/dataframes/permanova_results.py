# src/dataframes/permanova_results.py
import logging

import dataframely as dy

import bioregion_rs
from src.dataframes.geocode import GeocodeNoEdgesSchema
from src.dataframes.geocode_cluster import GeocodeClusterSchema
from src.matrices.geocode_distance import GeocodeDistanceMatrix

logger = logging.getLogger(__name__)


class PermanovaResultsSchema(dy.Schema):
    """
    Stores the results of a PERMANOVA test.
    """

    method_name = dy.String(nullable=False)  # e.g., "PERMANOVA"
    test_statistic_name = dy.String(nullable=False)  # e.g., "pseudo-F"
    test_statistic = dy.Float64(nullable=False)  # The calculated statistic value
    p_value = dy.Float64(nullable=False)  # The p-value of the test
    permutations = dy.UInt64(nullable=False)  # Number of permutations used


def build_permanova_results_df(
    geocode_distance_matrix: GeocodeDistanceMatrix,
    geocode_cluster_df: dy.DataFrame[GeocodeClusterSchema],
    geocode_lf: dy.LazyFrame[GeocodeNoEdgesSchema],
    permutations: int = 999,  # Default permutations
) -> dy.DataFrame[PermanovaResultsSchema]:
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

    return PermanovaResultsSchema.validate(df)
