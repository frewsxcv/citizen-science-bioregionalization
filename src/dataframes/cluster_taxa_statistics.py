from typing import Dict, List, Optional

import polars as pl

from src.dataframes.geocode_cluster import GeocodeClusterDataFrame
from src.dataframes.geocode_taxa_counts import GeocodeTaxaCountsDataFrame
from src.dataframes.taxonomy import TaxonomyDataFrame
from src.darwin_core import kingdom_enum
from src.types import ClusterId
from src.data_container import DataContainer, assert_dataframe_schema


class ClusterTaxaStatisticsDataFrame(DataContainer):
    df: pl.DataFrame
    SCHEMA = {
        "cluster": pl.UInt32(),  # `null` if stats for all clusters
        "kingdom": kingdom_enum,
        "taxonRank": pl.String(),
        "scientificName": pl.String(),
        "count": pl.UInt32(),
        "average": pl.Float64(),  # average of taxa with `name` within `rank` and `cluster`
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

        # Schema:
        #   - geocode: String
        #   - kingdom: Enum
        #   - taxonRank: String
        #   - scientificName: String
        #   - count: UInt32
        #   - phylum: String
        #   - class: String
        #   - order: String
        #   - family: String
        #   - genus: String
        #   - species: String
        joined = geocode_taxa_counts_dataframe.df.join(
            taxonomy_dataframe.df, on=["kingdom", "scientificName", "taxonRank"]
        )
        assert_dataframe_schema(
            joined,
            {
                "geocode": pl.String(),
                "kingdom": kingdom_enum,
                "taxonRank": pl.String(),
                "scientificName": pl.String(),
                "count": pl.UInt32(),
                "phylum": pl.String(),
                "class": pl.String(),
                "order": pl.String(),
                "family": pl.String(),
                "genus": pl.String(),
                "species": pl.String(),
            },
        )

        # Total count of all observations
        total_count = joined["count"].sum()

        # Calculate stats for all clusters
        df.vstack(
            joined.group_by(["kingdom", "scientificName", "taxonRank"])
            .agg(
                pl.col("count").sum().alias("count"),
                (pl.col("count").sum() / total_count).alias("average"),
            )
            .pipe(add_cluster_column, value=None)
            .select(cls.SCHEMA.keys()),  # Reorder columns
            in_place=True,
        )

        # Create a mapping from geocode to cluster
        geocode_to_cluster = geocode_cluster_dataframe.df.select(
            ["geocode", "cluster"]
        )
        
        # Join the cluster information with the data
        joined_with_cluster = joined.join(
            geocode_to_cluster,
            on="geocode",
            how="inner"
        )
        
        # Calculate total counts per cluster
        cluster_totals = (
            joined_with_cluster
            .group_by("cluster")
            .agg(
                pl.col("count").sum().alias("total_count_in_cluster")
            )
        )
        
        # Calculate stats for each cluster in one operation
        cluster_stats = (
            joined_with_cluster
            .group_by(["cluster", "kingdom", "taxonRank", "scientificName"])
            .agg(
                pl.col("count").sum().alias("count"),
            )
            .join(
                cluster_totals,
                on="cluster"
            )
            .with_columns([
                (pl.col("count") / pl.col("total_count_in_cluster")).alias("average")
            ])
            .drop("total_count_in_cluster")
            .select(cls.SCHEMA.keys())  # Ensure columns are in the right order
        )
        
        # Add cluster-specific stats to the dataframe
        df.vstack(cluster_stats, in_place=True)

        return cls(df=df)

    def _get_count_by_rank_and_name(self, rank: str, name: str) -> int:
        counts = self.df.filter(
            (pl.col("taxonRank") == rank) & (pl.col("scientificName") == name)
        ).get_column("count")
        assert len(counts) <= 1
        sum = counts.sum()
        assert isinstance(sum, int)
        return sum

    def order_count(self, order: str) -> int:
        return self._get_count_by_rank_and_name("order", order)

    def class_count(self, class_name: str) -> int:
        return self._get_count_by_rank_and_name("class", class_name)

    def waterfowl_count(self) -> int:
        return (
            self.order_count("Anseriformes")
            + self.order_count("Charadriiformes")
            + self.order_count("Gaviiformes")
        )

    def aves_count(self) -> int:
        return self.class_count("Aves")


def add_cluster_column(df: pl.DataFrame, value: Optional[int]) -> pl.DataFrame:
    return df.with_columns(pl.lit(value).cast(pl.UInt32()).alias("cluster"))
