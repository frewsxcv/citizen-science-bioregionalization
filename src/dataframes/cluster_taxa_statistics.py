from typing import Optional
import polars as pl
import dataframely as dy

from src.dataframes.geocode_cluster import GeocodeClusterSchema
from src.dataframes.geocode_taxa_counts import GeocodeTaxaCountsSchema
from src.dataframes.taxonomy import TaxonomyDataFrame
from src.types import ClusterId


class ClusterTaxaStatisticsSchema(dy.Schema):
    cluster = dy.UInt32(nullable=True)  # `null` if stats for all clusters
    taxonId = dy.UInt32(nullable=False)
    count = dy.UInt32(nullable=False)
    average = dy.Float64(
        nullable=False
    )  # average of taxa with `taxonId` within `cluster`

    @classmethod
    def build(
        cls,
        geocode_taxa_counts_dataframe: dy.DataFrame[GeocodeTaxaCountsSchema],
        geocode_cluster_dataframe: dy.DataFrame[GeocodeClusterSchema],
        taxonomy_dataframe: TaxonomyDataFrame,
    ) -> dy.DataFrame["ClusterTaxaStatisticsSchema"]:
        df = pl.DataFrame()

        # First, join the geocode_taxa_counts with taxonomy to get back the taxonomic info
        joined = geocode_taxa_counts_dataframe.join(
            taxonomy_dataframe.df, on="taxonId"
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
            .select(cls.columns().keys()),  # Reorder columns
            in_place=True,
        )

        # Create a mapping from geocode to cluster
        geocode_to_cluster = geocode_cluster_dataframe.select(["geocode", "cluster"])

        # Join the cluster information with the data
        joined_with_cluster = joined.join(geocode_to_cluster, on="geocode", how="inner")

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
            .select(cls.columns().keys())  # Ensure columns are in the right order
        )

        # Add cluster-specific stats to the dataframe
        df.vstack(cluster_stats, in_place=True)

        return cls.validate(df)


def iter_cluster_ids(
    cluster_taxa_statistics_dataframe: dy.DataFrame[ClusterTaxaStatisticsSchema],
) -> list[ClusterId]:
    return cluster_taxa_statistics_dataframe["cluster"].unique().to_list()


def add_cluster_column(df: pl.DataFrame, value: Optional[int]) -> pl.DataFrame:
    return df.with_columns(cluster=pl.lit(value).cast(pl.UInt32()))
