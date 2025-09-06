from typing import Dict, List, Optional, Self

import polars as pl

from src.dataframes.geocode_cluster import GeocodeClusterDataFrame
from src.dataframes.geocode_taxa_counts import GeocodeTaxaCountsDataFrame
from src.dataframes.taxonomy import TaxonomyDataFrame
from polars_darwin_core.darwin_core import kingdom_data_type
from src.types import ClusterId
from src.data_container import DataContainer, assert_dataframe_schema


class ClusterTaxaCountsDataFrame(DataContainer):
    df: pl.DataFrame
    SCHEMA = {
        "cluster": pl.UInt32(),  # `null` if counts for all clusters
        "kingdom": kingdom_data_type,
        "taxonRank": pl.String(),
        "scientificName": pl.String(),
        "count": pl.UInt32(),
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
    ) -> Self:
        df = pl.DataFrame(schema=cls.SCHEMA)

        # First, join the geocode_taxa_counts with taxonomy to get back the taxonomic info
        joined = geocode_taxa_counts_dataframe.df.join(
            taxonomy_dataframe.df, on="taxonId"
        )

        # Verify the schema of the joined dataframe
        assert_dataframe_schema(
            joined,
            {
                "geocode": pl.Uint64(),
                "taxonId": pl.UInt32(),
                "count": pl.UInt32(),
                "kingdom": kingdom_data_type,
                "phylum": pl.String(),
                "class": pl.String(),
                "order": pl.String(),
                "family": pl.String(),
                "genus": pl.String(),
                "species": pl.String(),
                "taxonRank": pl.String(),
                "scientificName": pl.String(),
            },
        )

        # Calculate counts for all clusters combined
        df.vstack(
            joined.group_by(["kingdom", "scientificName", "taxonRank"])
            .agg(pl.col("count").sum().alias("count"))
            .pipe(add_cluster_column, value=None)
            .select(cls.SCHEMA.keys()),  # Reorder columns
            in_place=True,
        )

        # Create a mapping from geocode to cluster
        geocode_to_cluster = geocode_cluster_dataframe.df.select(["geocode", "cluster"])

        # Join the cluster information with the data
        joined_with_cluster = joined.join(geocode_to_cluster, on="geocode", how="inner")

        # Calculate counts for each cluster
        cluster_counts = (
            joined_with_cluster.group_by(
                ["cluster", "kingdom", "taxonRank", "scientificName"]
            )
            .agg(pl.col("count").sum().alias("count"))
            .select(cls.SCHEMA.keys())  # Ensure columns are in the right order
        )

        # Add cluster-specific counts to the dataframe
        df.vstack(cluster_counts, in_place=True)

        return cls(df=df)

    def _get_count_by_rank_and_name(self, rank: str, name: str) -> int:
        counts = self.df.filter(
            (pl.col("taxonRank") == rank) & (pl.col("scientificName") == name)
        ).get_column("count")
        assert len(counts) <= 1
        sum = counts.sum()
        assert isinstance(sum, int)
        return sum

    def filter_by_cluster(self, cluster_id: ClusterId) -> Self:
        """
        Returns a new dataframe filtered to only include data for the specified cluster.
        """
        filtered_df = self.df.filter(pl.col("cluster") == cluster_id)
        return self.__class__(df=filtered_df)


def add_cluster_column(df: pl.DataFrame, value: Optional[int]) -> pl.DataFrame:
    return df.with_columns(pl.lit(value).cast(pl.UInt32()).alias("cluster"))
