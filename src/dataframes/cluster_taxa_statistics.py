import logging

import dataframely as dy

import bioregion_rs
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

    return ClusterTaxaStatisticsSchema.validate(df)


def iter_cluster_ids(
    cluster_taxa_statistics_df: dy.DataFrame[ClusterTaxaStatisticsSchema],
) -> list[ClusterId]:
    return cluster_taxa_statistics_df["cluster"].unique().to_list()
