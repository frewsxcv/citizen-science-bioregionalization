import polars as pl
from src.dataframes.geocode_cluster import GeocodeClusterDataFrame
from src.dataframes.permanova_results import PermanovaResultsDataFrame
import logging

logger = logging.getLogger(__name__)


def _format_permanova_results(results_df: PermanovaResultsDataFrame) -> str:
    """Formats the PERMANOVA results DataFrame into a readable string."""
    if results_df.df.height == 0 or results_df.df["test_statistic"].is_null()[0]:
        # Handle case where PERMANOVA might not have run successfully upstream
        # (though build() should ideally raise an error now)
        return "PERMANOVA results not available."

    row = results_df.df.row(0, named=True) # Get the first (only) row as a dict

    # Handle cases where permutations might be 0 or None
    perms = row['permutations']
    perms_str = str(perms) if perms is not None and perms > 0 else "N/A"

    return (
        f"--- {row['method_name']} Results ---\n"
        f"Permutations: {perms_str}\n"
        f"{row['test_statistic_name']} statistic: {row['test_statistic']:.4f}\n"
        f"P-value: {row['p_value']:.4f}\n"
        f"------------------------------"
    )


def print_results(
    geocode_cluster_dataframe: GeocodeClusterDataFrame,
    permanova_results_dataframe: PermanovaResultsDataFrame,
) -> None:

    # Print PERMANOVA results using the local formatting function
    print("\n" + _format_permanova_results(permanova_results_dataframe) + "\n")

    # Log number of clusters found
    logger.info(f"Number of clusters found: {geocode_cluster_dataframe.num_clusters()}")
    print("-" * 25) # Separator
