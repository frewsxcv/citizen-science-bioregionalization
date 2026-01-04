import logging
from typing import Dict, TypeVar, Union, cast

import dataframely as dy
import polars as pl
import seaborn as sns
from scipy.cluster.hierarchy import linkage

from src.dataframes.cluster_color import ClusterColorSchema, get_color_for_cluster
from src.dataframes.cluster_significant_differences import (
    ClusterSignificantDifferencesSchema,
)
from src.dataframes.cluster_taxa_statistics import ClusterTaxaStatisticsSchema
from src.dataframes.geocode import GeocodeNoEdgesSchema
from src.dataframes.geocode_cluster import GeocodeClusterSchema, cluster_for_geocode
from src.dataframes.geocode_taxa_counts import GeocodeTaxaCountsSchema

NumericSeries = TypeVar("NumericSeries", bound=pl.Series)

logger = logging.getLogger(__name__)


def create_cluster_taxa_heatmap(
    geocode_lf: dy.LazyFrame[GeocodeNoEdgesSchema],
    geocode_cluster_df: dy.DataFrame[GeocodeClusterSchema],
    cluster_colors_df: dy.DataFrame["ClusterColorSchema"],
    geocode_distance_matrix,
    cluster_significant_differences_df: dy.DataFrame[
        "ClusterSignificantDifferencesSchema"
    ],
    taxonomy_df,
    geocode_taxa_counts_lf: dy.LazyFrame[GeocodeTaxaCountsSchema],
    cluster_taxa_statistics_df: dy.DataFrame[ClusterTaxaStatisticsSchema],
    limit_species=None,
):
    logger.info("create_cluster_taxa_heatmap: Starting")

    # Log input dataframe sizes
    geocode_lf_collected = geocode_lf.collect(engine="streaming")
    logger.info(f"  geocode_lf: {geocode_lf_collected.height} rows")

    geocode_cluster_df_height = geocode_cluster_df.height
    logger.info(f"  geocode_cluster_df: {geocode_cluster_df_height} rows")

    cluster_colors_df_height = cluster_colors_df.height
    logger.info(f"  cluster_colors_df: {cluster_colors_df_height} rows")

    geocode_taxa_counts_collected = geocode_taxa_counts_lf.collect(engine="streaming")
    logger.info(
        f"  geocode_taxa_counts_lf: {geocode_taxa_counts_collected.height} rows"
    )

    cluster_taxa_statistics_df_height = cluster_taxa_statistics_df.height
    logger.info(
        f"  cluster_taxa_statistics_df: {cluster_taxa_statistics_df_height} rows"
    )

    cluster_significant_differences_df_height = (
        cluster_significant_differences_df.height
    )
    logger.info(
        f"  cluster_significant_differences_df: {cluster_significant_differences_df_height} rows"
    )

    taxonomy_df_height = taxonomy_df.height
    logger.info(f"  taxonomy_df: {taxonomy_df_height} rows")

    # Use only geocodes that exist in both geocode_lf AND geocode_taxa_counts_lf
    # This ensures we don't try to compute averages for geocodes with no taxa data
    geocodes_in_geocode_lf = geocode_lf_collected.select("geocode").unique()
    logger.info(f"  Unique geocodes in geocode_lf: {geocodes_in_geocode_lf.height}")

    geocodes_with_taxa = geocode_taxa_counts_collected.select("geocode").unique()
    logger.info(
        f"  Unique geocodes in geocode_taxa_counts_lf: {geocodes_with_taxa.height}"
    )

    ordered_geocodes = geocodes_in_geocode_lf.join(
        geocodes_with_taxa, on="geocode", how="semi"
    ).to_series()
    logger.info(
        f"  Geocodes after intersection (ordered_geocodes): {len(ordered_geocodes)}"
    )

    # Check for geocodes in geocode_lf that are NOT in geocode_taxa_counts_lf
    geocodes_without_taxa = geocodes_in_geocode_lf.join(
        geocodes_with_taxa, on="geocode", how="anti"
    )
    if geocodes_without_taxa.height > 0:
        logger.warning(
            f"  WARNING: {geocodes_without_taxa.height} geocodes in geocode_lf have no taxa data!"
        )
        logger.warning(
            f"  Geocodes without taxa: {geocodes_without_taxa['geocode'].to_list()[:10]}..."
        )

    # Create color mapping for geocodes by cluster
    col_colors = []
    for geocode in ordered_geocodes:
        cluster = cluster_for_geocode(geocode_cluster_df, geocode)
        col_colors.append(get_color_for_cluster(cluster_colors_df, cluster))

    # Compute linkage for clustering
    linkage_array = linkage(geocode_distance_matrix.condensed(), "ward")

    # Join taxonomic information
    joined = cluster_significant_differences_df.join(
        taxonomy_df, on=["taxonId"], how="left"
    )
    logger.info(f"  Joined significant_differences with taxonomy: {joined.height} rows")

    # Process data for each species/taxon
    data = {}
    species_query = joined.select("scientificName", "taxonId").unique()
    logger.info(f"  Unique species to process: {species_query.height}")

    # If there are no significant differences, return None
    if species_query.height == 0:
        logger.info("  No significant differences found, returning None")
        return None

    # Apply limit if specified
    if limit_species is not None:
        species_query = species_query.limit(limit_species)
        logger.info(f"  Limited to {limit_species} species")

    logger.info(
        f"  Processing {species_query.height} species across {len(ordered_geocodes)} geocodes"
    )

    for species, taxonId in species_query.iter_rows():
        counts = []
        logger.info(f"  Processing species: {species} (taxonId={taxonId})")

        for geocode in ordered_geocodes:
            # Use collected DataFrame instead of LazyFrame to avoid re-evaluation
            # issues and improve performance (no repeated collection per iteration)
            geocode_counts_species = (
                geocode_taxa_counts_collected.filter(
                    pl.col("geocode") == geocode, pl.col("taxonId") == taxonId
                )
                .select("count")
                .sum()
                .item()
            )
            geocode_counts_all = (
                geocode_taxa_counts_collected.filter(pl.col("geocode") == geocode)
                .select("count")
                .sum()
                .item()
            )

            if geocode_counts_all is None or geocode_counts_all == 0:
                logger.error(
                    f"  DIVISION BY ZERO: geocode={geocode}, taxonId={taxonId}, "
                    f"geocode_counts_species={geocode_counts_species}, geocode_counts_all={geocode_counts_all}"
                )
                raise ZeroDivisionError(
                    f"geocode_counts_all is {geocode_counts_all} for geocode={geocode}. "
                    f"This geocode should have been filtered out but wasn't."
                )

            geocode_average = geocode_counts_species / geocode_counts_all
            all_average = (
                cluster_taxa_statistics_df.filter(
                    pl.col("taxonId") == taxonId,
                    pl.col("cluster").is_null(),
                )
                .get_column("average")
                .item()
            )
            counts.append(geocode_average - all_average)
        counts = pl.Series(
            values=counts,
            name=species,
        )
        # We can safely cast the Series to a numeric series since we know it contains
        # numeric values resulting from mathematical operations
        data[species] = min_max_normalize(counts)

    logger.info(f"  Finished processing all species, creating clustermap")

    # Create dataframe and generate clustermap
    dataframe = pl.DataFrame(data=data)
    logger.info(f"  Heatmap dataframe shape: {dataframe.shape}")

    g = sns.clustermap(
        data=dataframe,  # type: ignore
        col_cluster=False,
        row_cluster=True,
        row_linkage=linkage_array,  # type: ignore
        row_colors=col_colors,
        xticklabels=dataframe.columns,
        yticklabels=False,
    )

    return g


def min_max_normalize(series: pl.Series) -> pl.Series:
    """
    Normalize a numeric series to range [0, 1] using min-max normalization.

    Parameters:
    -----------
    series: pl.Series
        The numeric series to normalize. Must contain numeric values.

    Returns:
    --------
    pl.Series
        The normalized series with values between 0 and 1
    """
    # Since mypy cannot infer operations on pl.Series properly,
    # we proceed with the implementation knowing the series contains numeric values
    min_val = series.min()
    max_val = series.max()

    # Handle the case where min and max are the same (constant series)
    if min_val == max_val:
        return pl.Series(values=[0.5] * len(series), name=series.name)

    # This is a numerical operation that's valid for numeric series
    # but mypy can't verify it properly
    return (series - min_val) / (max_val - min_val)  # type: ignore
