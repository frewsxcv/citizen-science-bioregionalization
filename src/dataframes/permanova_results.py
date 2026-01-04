# src/dataframes/permanova_results.py
import logging

import dataframely as dy
import pandas as pd  # Import pandas for type hint
import polars as pl
from skbio.stats.distance import DistanceMatrix, permanova

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

    # Create the skbio DistanceMatrix object. Let ValueError propagate if IDs mismatch.
    # Collect the LazyFrame once at the start
    geocode_df = geocode_lf.collect()
    geocode_ids = geocode_df["geocode"].to_list()
    dm_skbio = DistanceMatrix(geocode_distance_matrix.condensed(), ids=geocode_ids)

    # Assert that all geocodes in the distance matrix have cluster assignments.
    cluster_geocodes = set(geocode_cluster_df["geocode"].to_list())
    missing_geocodes = set(geocode_ids) - cluster_geocodes
    assert not missing_geocodes, (
        f"Missing cluster assignments for {len(missing_geocodes)} geocodes required by the distance matrix: {missing_geocodes}"
    )

    # Get cluster assignments in the correct order matching the distance matrix.
    # Join geocode_lf (which defines the order) with cluster assignments.
    # Inner join is safe because the assertion passed.
    grouping_df = geocode_df.join(
        geocode_cluster_df.select(["geocode", "cluster"]),
        on="geocode",
        how="inner",  # Should match all rows due to assertion
    )
    # The join preserves the order of the left dataframe (geocode_lf)
    grouping = grouping_df["cluster"].to_list()

    # Verify lengths match (should always pass due to assertion and join logic)
    assert len(grouping) == len(dm_skbio.ids), (
        f"Internal error: Mismatch between grouping length ({len(grouping)}) and distance matrix IDs length ({len(dm_skbio.ids)}) after alignment."
    )

    # Run PERMANOVA. Let exceptions propagate.
    results: pd.Series = permanova(dm_skbio, grouping, permutations=permutations)

    # Extract results into a Polars DataFrame
    results_dict = {
        "method_name": ["PERMANOVA"],
        "test_statistic_name": ["pseudo-F"],
        "test_statistic": [results.get("test statistic")],
        "p_value": [results.get("p-value")],
        "permutations": [results.get("permutations")],
    }
    results_df = pl.DataFrame(results_dict).with_columns(
        pl.col("permutations").cast(pl.UInt64).fill_null(0)
    )

    return PermanovaResultsSchema.validate(results_df)
