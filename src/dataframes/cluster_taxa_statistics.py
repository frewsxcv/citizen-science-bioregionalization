from typing import Dict, List, Optional

import polars as pl

from src.dataframes.geocode_cluster import GeocodeClusterDataFrame
from src.dataframes.geocode_taxa_counts import GeocodeTaxaCountsDataFrame
from src.dataframes.taxonomy import TaxonomyDataFrame
from src.types import ClusterId
from src.data_container import DataContainer, assert_dataframe_schema


class ClusterTaxaStatisticsDataFrame(DataContainer):
    df: pl.DataFrame
    SCHEMA = {
        "cluster": pl.UInt32(),  # `null` if stats for all clusters
        "taxonId": pl.UInt32(),
        "count": pl.UInt32(),
        "average": pl.Float64(),  # average of taxa with `taxonId` within `cluster`
    }

    def __init__(self, df: pl.DataFrame) -> None:
        assert_dataframe_schema(df, self.SCHEMA)
        self.df = df

    def iter_cluster_ids(self) -> list[ClusterId]:
        return self.df["cluster"].unique().to_list()

    @classmethod
    def build(
        cls,
        geocode_taxa_counts_dataframe: GeocodeTaxaCountsDataFrame,
        geocode_cluster_dataframe: GeocodeClusterDataFrame,
        taxonomy_dataframe: TaxonomyDataFrame,
    ) -> "ClusterTaxaStatisticsDataFrame":
        df = pl.DataFrame(schema=cls.SCHEMA)

        # First, join the geocode_taxa_counts with taxonomy to get back the taxonomic info
        joined = geocode_taxa_counts_dataframe.df.join(
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
            .select(cls.SCHEMA.keys()),  # Reorder columns
            in_place=True,
        )

        # Create a mapping from geocode to cluster
        geocode_to_cluster = geocode_cluster_dataframe.df.select(["geocode", "cluster"])

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
            .select(cls.SCHEMA.keys())  # Ensure columns are in the right order
        )

        # Add cluster-specific stats to the dataframe
        df.vstack(cluster_stats, in_place=True)

        return cls(df=df)


def add_cluster_column(df: pl.DataFrame, value: Optional[int]) -> pl.DataFrame:
    return df.with_columns(pl.lit(value).cast(pl.UInt32()).alias("cluster"))
