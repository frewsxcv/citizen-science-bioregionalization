import polars as pl
from src.darwin_core import TaxonRank
from src.dataframes.cluster_taxa_statistics import ClusterTaxaStatisticsDataFrame
from src.data_container import DataContainer
from typing import Self


class ClusterSignificantDifferencesDataFrame(DataContainer):
    """
    A dataframe that contains the significant differences between clusters.
    """

    THRESHOLD = 10  # Percent difference

    SCHEMA = {
        "cluster": pl.UInt32,
        "taxon": pl.String,
        "percentage_difference": pl.Float64,  # TODO: should this a p-value?
    }

    def __init__(self, df: pl.DataFrame):
        self.df = df

    @classmethod
    def build(cls, all_stats: ClusterTaxaStatisticsDataFrame) -> Self:
        # Calculate significant differences
        significant_differences = []

        for cluster in all_stats.iter_cluster_ids():
            for kingdom, species, average in (
                all_stats.df.filter(
                    pl.col("cluster") == cluster,
                    pl.col("rank") == TaxonRank.species,
                )
                .sort(by="count", descending=True)
                .limit(20)  # TODO: Does this need to happen?
                .select(["kingdom", "name", "average"])
                .iter_rows(named=False)
            ):
                all_average = (
                    all_stats.df.filter(
                        pl.col("kingdom") == kingdom,
                        pl.col("name") == species,
                        pl.col("cluster").is_null(),
                        pl.col("rank") == TaxonRank.species,
                    )
                    .get_column("average")
                    .item()
                )

                percent_diff = (average / all_average * 100) - 100
                if abs(percent_diff) > cls.THRESHOLD:
                    significant_differences.append(
                        {
                            "cluster": cluster,
                            "taxon": species,
                            "percentage_difference": percent_diff,
                        }
                    )

        # Create a DataFrame from the significant differences
        df = pl.DataFrame(significant_differences, schema=cls.SCHEMA)
        return cls(df)
