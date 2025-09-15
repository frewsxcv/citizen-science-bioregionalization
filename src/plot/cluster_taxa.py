import seaborn as sns
import polars as pl
from typing import Dict, Union, TypeVar, cast
from scipy.cluster.hierarchy import linkage

import dataframely as dy
from src.dataframes.cluster_color import ClusterColorSchema, get_color_for_cluster
from src.dataframes.cluster_significant_differences import (
    ClusterSignificantDifferencesSchema,
)
from src.dataframes.geocode_cluster import GeocodeClusterSchema, cluster_for_geocode
from src.dataframes.cluster_taxa_statistics import ClusterTaxaStatisticsSchema
NumericSeries = TypeVar("NumericSeries", bound=pl.Series)


def create_cluster_taxa_heatmap(
    geocode_dataframe,
    geocode_cluster_dataframe: dy.DataFrame[GeocodeClusterSchema],
    cluster_colors_dataframe: dy.DataFrame["ClusterColorSchema"],
    geocode_distance_matrix,
    cluster_significant_differences_dataframe: dy.DataFrame[
        "ClusterSignificantDifferencesSchema"
    ],
    taxonomy_dataframe,
    geocode_taxa_counts_dataframe,
    cluster_taxa_statistics_dataframe: dy.DataFrame[ClusterTaxaStatisticsSchema],
    limit_species=None,
):
    """
    Create a heatmap visualization of taxa distributions across geocodes.

    Parameters:
    -----------
    geocode_dataframe: DataContainer
        The dataframe containing geocode data
    geocode_cluster_dataframe: DataContainer
        The dataframe with geocode-to-cluster mappings
    cluster_colors_dataframe: DataContainer
        The dataframe with cluster color information
    geocode_distance_matrix: GeocodeDistanceMatrix
        The distance matrix between geocodes
    cluster_significant_differences_dataframe: DataContainer
        The dataframe with significant taxonomic differences
    taxonomy_dataframe: DataContainer
        The dataframe with taxonomy information
    geocode_taxa_counts_dataframe: DataContainer
        The dataframe with taxa counts per geocode
    cluster_taxa_statistics_dataframe: DataContainer
        The dataframe with taxa statistics per cluster
    limit_species: int, optional
        If provided, limit the number of species shown in the heatmap

    Returns:
    --------
    g: seaborn.ClusterGrid
        The resulting clustermap visualization
    """
    ordered_geocodes = geocode_dataframe["geocode"].unique()

    # Create color mapping for geocodes by cluster
    col_colors = []
    for geocode in ordered_geocodes:
        cluster = cluster_for_geocode(geocode_cluster_dataframe, geocode)
        col_colors.append(get_color_for_cluster(cluster_colors_dataframe, cluster))

    # Compute linkage for clustering
    linkage_array = linkage(geocode_distance_matrix.condensed(), "ward")

    # Join taxonomic information
    joined = cluster_significant_differences_dataframe.join(
        taxonomy_dataframe, on=["taxonId"], how="left"
    )

    # Process data for each species/taxon
    data = {}
    species_query = joined.select("scientificName", "taxonId").unique()

    # Apply limit if specified
    if limit_species is not None:
        species_query = species_query.limit(limit_species)

    for species, taxonId in species_query.iter_rows():
        counts = []

        for geocode in ordered_geocodes:
            geocode_counts_species = (
                geocode_taxa_counts_dataframe.lazy()
                .filter(pl.col("geocode") == geocode, pl.col("taxonId") == taxonId)
                .select("count")
                .sum()
                .collect()
                .item()
            )
            geocode_counts_all = (
                geocode_taxa_counts_dataframe.lazy()
                .filter(pl.col("geocode") == geocode)
                .select("count")
                .sum()
                .collect()
                .item()
            )
            geocode_average = geocode_counts_species / geocode_counts_all
            all_average = (
                cluster_taxa_statistics_dataframe.filter(
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

    # Create dataframe and generate clustermap
    dataframe = pl.DataFrame(data=data)
    g = sns.clustermap(
        data=dataframe, # type: ignore
        col_cluster=False,
        row_cluster=True,
        row_linkage=linkage_array, # type: ignore
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
