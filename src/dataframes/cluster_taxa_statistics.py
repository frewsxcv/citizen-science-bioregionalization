import polars as pl
import logging


import bioregion_rs

logger = logging.getLogger(__name__)

def build_cluster_taxa_statistics_df(
    geocode_taxa_counts_lf: pl.LazyFrame,
    geocode_cluster_lf: pl.LazyFrame,
    taxonomy_lf: pl.LazyFrame,
) -> pl.DataFrame:
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

    cluster_df = geocode_cluster_lf.select("geocode", "cluster").collect(
        engine="streaming"
    )
    cluster_geocodes = cluster_df.select("geocode").unique().height
    cluster_clusters = cluster_df.select("cluster").unique().height
    logger.info(
        f"build_cluster_taxa_statistics_df: geocode_cluster_lf has {cluster_df.height} rows, "
        f"{cluster_geocodes} unique geocodes, {cluster_clusters} unique clusters"
    )

    taxonomy_df = taxonomy_lf.select("taxonId").collect(engine="streaming")

    df = bioregion_rs.build_cluster_taxa_statistics(
        taxa_counts_df, cluster_df, taxonomy_df
    )

    logger.info(f"build_cluster_taxa_statistics_df: Final output has {df.height} rows")

    return df
