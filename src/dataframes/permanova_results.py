# src/dataframes/permanova_results.py
from typing import Any
import polars as pl
import pandas as pd # Import pandas for type hint
from skbio.stats.distance import permanova, DistanceMatrix
from src.data_container import DataContainer, assert_dataframe_schema
from src.dataframes.geocode_cluster import GeocodeClusterDataFrame
from src.matrices.geocode_distance import GeocodeDistanceMatrix
from src.dataframes.geocode import GeocodeDataFrame
import logging

logger = logging.getLogger(__name__)

class PermanovaResultsDataFrame(DataContainer):
    """
    Stores the results of a PERMANOVA test.

    Attributes:
        df (pl.DataFrame): DataFrame containing PERMANOVA results.
                           Expected columns match the SCHEMA.
    """
    df: pl.DataFrame
    SCHEMA = {
        "method_name": pl.Utf8(),      # e.g., "PERMANOVA"
        "test_statistic_name": pl.Utf8(), # e.g., "pseudo-F"
        "test_statistic": pl.Float64(), # The calculated statistic value
        "p_value": pl.Float64(),        # The p-value of the test
        "permutations": pl.UInt64(),    # Number of permutations used
        # Potentially add degrees of freedom if needed/available
    }

    def __init__(self, df: pl.DataFrame) -> None:
        assert_dataframe_schema(df, self.SCHEMA)
        self.df = df

    @classmethod
    def build(
        cls,
        geocode_distance_matrix: GeocodeDistanceMatrix,
        geocode_cluster_dataframe: GeocodeClusterDataFrame,
        geocode_dataframe: GeocodeDataFrame,
        permutations: int = 999, # Default permutations
    ) -> "PermanovaResultsDataFrame":
        """
        Runs the PERMANOVA test and stores the results.

        Args:
            geocode_distance_matrix: Wrapper containing the condensed distance matrix.
            geocode_cluster_dataframe: DataFrame mapping geocodes to clusters.
            geocode_dataframe: DataFrame containing the ordered list of geocodes corresponding
                               to the distance matrix.
            permutations: Number of permutations for the test.

        Returns:
            An instance of PermanovaResultsDataFrame.
        """
        # Create the skbio DistanceMatrix object
        # The order of IDs must match the order used when creating the condensed matrix
        geocode_ids = geocode_dataframe.df["geocode"].to_list()
        try:
            dm_skbio = DistanceMatrix(geocode_distance_matrix.condensed(), ids=geocode_ids)
        except ValueError as e:
             logger.error(f"Failed to create skbio DistanceMatrix: {e}. Ensure geocode IDs match condensed matrix size.")
             # Return empty dataframe if DistanceMatrix creation fails
             return cls(df=pl.DataFrame(schema=cls.SCHEMA))


        # Ensure cluster assignments align with the distance matrix IDs
        # Get the cluster grouping for each geocode present in the distance matrix
        dm_ids = dm_skbio.ids
        cluster_mapping = geocode_cluster_dataframe.df.filter(pl.col("geocode").is_in(list(dm_ids)))

        # Check if all geocodes in the distance matrix have a cluster assignment
        if len(cluster_mapping) != len(dm_ids):
             missing_ids = set(dm_ids) - set(cluster_mapping['geocode'].to_list())
             logger.warning(f"Missing cluster assignments for {len(missing_ids)} geocodes in distance matrix. Filtering them out for PERMANOVA.")
             # Filter the distance matrix to only include geocodes with cluster assignments
             dm_filtered = dm_skbio.filter(cluster_mapping['geocode'].to_list())
             # Re-fetch the cluster mapping for the filtered IDs, ensuring correct order
             cluster_mapping = cluster_mapping.filter(pl.col("geocode").is_in(dm_filtered.ids))
             dm_ids = dm_filtered.ids # Update dm_ids to the filtered list
             dm_to_use = dm_filtered
        else:
             dm_to_use = dm_skbio

        # Create a DataFrame for sorting based on the distance matrix ID order
        sort_order_df = pl.DataFrame({
            "geocode": list(dm_to_use.ids),
            "sort_order": range(len(dm_to_use.ids))
        })

        # Join cluster mapping with sort order and sort
        grouping_df = cluster_mapping.join(
            sort_order_df, on="geocode", how="inner"
        ).sort("sort_order")

        grouping = grouping_df["cluster"].to_list()


        # Verify lengths match after potential filtering and sorting
        if len(grouping) != len(dm_to_use.ids):
             raise ValueError(f"Mismatch between grouping length ({len(grouping)}) and distance matrix IDs length ({len(dm_to_use.ids)}) after alignment.")


        # Run PERMANOVA
        try:
            results: pd.Series = permanova(dm_to_use, grouping, permutations=permutations)

            # Extract results into a Polars DataFrame using Series indexing
            results_dict = {
                "method_name": ["PERMANOVA"],
                "test_statistic_name": ["pseudo-F"], # Defaulting test statistic name
                "test_statistic": [results.get("test statistic")],
                "p_value": [results.get("p-value")],
                "permutations": [results.get("permutations")],
            }
            results_df = pl.DataFrame(results_dict)

            # Ensure schema types (Polars might infer differently)
            results_df = results_df.with_columns([
                pl.col("method_name").cast(cls.SCHEMA["method_name"]),
                pl.col("test_statistic_name").cast(cls.SCHEMA["test_statistic_name"]),
                pl.col("test_statistic").cast(cls.SCHEMA["test_statistic"]),
                pl.col("p_value").cast(cls.SCHEMA["p_value"]),
                pl.col("permutations").cast(cls.SCHEMA["permutations"]),
            ])
        except Exception as e:
             logger.error(f"PERMANOVA calculation failed: {e}")
             # Return an empty dataframe matching the schema if PERMANOVA fails
             results_df = pl.DataFrame(schema=cls.SCHEMA)


        return cls(df=results_df)

    def format_results(self) -> str:
        """Formats the PERMANOVA results into a readable string."""
        if self.df.height == 0 or self.df["test_statistic"].is_null()[0]: # Check if results are missing
            return "PERMANOVA results not available (calculation may have failed)."

        row = self.df.row(0, named=True) # Get the first (only) row as a dict

        # Handle cases where permutations might be 0 or None if test wasn't run properly
        perms = row['permutations']
        perms_str = str(perms) if perms is not None and perms > 0 else "N/A"

        return (
            f"--- {row['method_name']} Results ---\n"
            f"Permutations: {perms_str}\n"
            f"{row['test_statistic_name']} statistic: {row['test_statistic']:.4f}\n"
            f"P-value: {row['p_value']:.4f}\n"
            f"------------------------------"
        )