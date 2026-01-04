import logging
from typing import Optional

import dataframely as dy
import polars as pl

from src.dataframes.geocode_cluster import GeocodeClusterSchema
from src.dataframes.geocode_taxa_counts import GeocodeTaxaCountsSchema
from src.dataframes.taxonomy import TaxonomySchema
from src.types import ClusterId

logger = logging.getLogger(__name__)


class ClusterTaxaStatisticsSchema(dy.Schema):
    cluster = dy.UInt32(nullable=True)  # `null` if stats for all clusters
    taxonId = dy.UInt32(nullable=False)
    count = dy.UInt32(nullable=False)
    average = dy.Float64(
        nullable=False
    )  # average of taxa with `taxonId` within `cluster`


def build_cluster_taxa_statistics_df(
    geocode_taxa_counts_lf: dy.LazyFrame[GeocodeTaxaCountsSchema],
    geocode_cluster_lf: dy.LazyFrame[GeocodeClusterSchema],
    taxonomy_lf: dy.LazyFrame[TaxonomySchema],
) -> dy.DataFrame[ClusterTaxaStatisticsSchema]:
    """Build cluster taxa statistics from geocode taxa counts.

    Computes aggregated statistics (count, average) for each taxon within each cluster,
    as well as overall statistics across all clusters (cluster=null).

    Args:
        geocode_taxa_counts_lf: LazyFrame of taxa counts per geocode
        geocode_cluster_lf: LazyFrame mapping geocodes to clusters
        taxonomy_lf: LazyFrame of taxonomy information

    Returns:
        A validated DataFrame conforming to ClusterTaxaStatisticsSchema
    """
    logger.info("build_cluster_taxa_statistics_df: Starting")

    # Log input sizes
    taxa_counts_df = geocode_taxa_counts_lf.collect(engine="streaming")
    taxa_counts_geocodes = taxa_counts_df.select("geocode").unique().height
    taxa_counts_taxa = taxa_counts_df.select("taxonId").unique().height
    logger.info(
        f"build_cluster_taxa_statistics_df: geocode_taxa_counts_lf has {taxa_counts_df.height} rows, "
        f"{taxa_counts_geocodes} unique geocodes, {taxa_counts_taxa} unique taxa"
    )

    cluster_df = geocode_cluster_lf.collect(engine="streaming")
    cluster_geocodes = cluster_df.select("geocode").unique().height
    cluster_clusters = cluster_df.select("cluster").unique().height
    logger.info(
        f"build_cluster_taxa_statistics_df: geocode_cluster_lf has {cluster_df.height} rows, "
        f"{cluster_geocodes} unique geocodes, {cluster_clusters} unique clusters"
    )

    df = pl.DataFrame()

    # First, join the geocode_taxa_counts with taxonomy to get back the taxonomic info
    joined = geocode_taxa_counts_lf.join(taxonomy_lf, on="taxonId").collect(
        engine="streaming"
    )

    logger.info(
        f"build_cluster_taxa_statistics_df: After joining with taxonomy: {joined.height} rows"
    )

    # Total count of all observations
    total_count = joined["count"].sum()

    # Calculate stats for all clusters
    df.vstack(
        joined.group_by(["taxonId"])
        .agg(
            pl.col("count").sum().alias("count"),
            (pl.col("count").sum() / total_count).alias("average"),
        )
        .pipe(add_cluster_column, value=None)
        .select(ClusterTaxaStatisticsSchema.columns().keys()),  # Reorder columns
        in_place=True,
    )

    # Create a mapping from geocode to cluster
    geocode_to_cluster = geocode_cluster_lf.select(["geocode", "cluster"])

    # Join the cluster information with the data
    joined_with_cluster = joined.lazy().join(
        geocode_to_cluster, on="geocode", how="inner"
    )

    # Calculate total counts per cluster
    cluster_totals = joined_with_cluster.group_by("cluster").agg(
        pl.col("count").sum().alias("total_count_in_cluster")
    )

    # Calculate stats for each cluster in one operation
    cluster_stats = (
        joined_with_cluster.group_by(["cluster", "taxonId"])
        .agg(
            pl.col("count").sum().alias("count"),
        )
        .join(cluster_totals, on="cluster")
        .with_columns(
            [(pl.col("count") / pl.col("total_count_in_cluster")).alias("average")]
        )
        .drop("total_count_in_cluster")
        .select(
            ClusterTaxaStatisticsSchema.columns().keys()
        )  # Ensure columns are in the right order
        .collect(engine="streaming")
    )

    # Add cluster-specific stats to the dataframe
    df.vstack(cluster_stats, in_place=True)

    logger.info(f"build_cluster_taxa_statistics_df: Final output has {df.height} rows")

    return ClusterTaxaStatisticsSchema.validate(df)


def iter_cluster_ids(
    cluster_taxa_statistics_df: dy.DataFrame[ClusterTaxaStatisticsSchema],
) -> list[ClusterId]:
    return cluster_taxa_statistics_df["cluster"].unique().to_list()


def add_cluster_column(df: pl.DataFrame, value: Optional[int]) -> pl.DataFrame:
    return df.with_columns(cluster=pl.lit(value).cast(pl.UInt32()))
