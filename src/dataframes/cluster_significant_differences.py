import polars as pl
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
        "kingdom": pl.String,
        "taxonRank": pl.String,
        "scientificName": pl.String,
        "percentage_difference": pl.Float64,  # TODO: should this a p-value?
    }

    def __init__(self, df: pl.DataFrame):
        self.df = df

    @classmethod
    def build(cls, all_stats: ClusterTaxaStatisticsDataFrame) -> Self:
        # Calculate significant differences
        significant_differences = []

        for cluster in all_stats.iter_cluster_ids():
            for kingdom, taxonRank, scientificName, average in (
                all_stats.df.filter(
                    (
                        pl.col("cluster").is_null()
                        if cluster is None
                        else pl.col("cluster") == cluster
                    ),
                )
                .sort(by="count", descending=True)
                .limit(20)  # TODO: Does this need to happen?
                .select(["kingdom", "taxonRank", "scientificName", "average"])
                .iter_rows(named=False)
            ):
                all_average = (
                    all_stats.df.filter(
                        pl.col("kingdom") == kingdom,
                        pl.col("taxonRank") == taxonRank,
                        pl.col("scientificName") == scientificName,
                        pl.col("cluster").is_null(),
                    )
                    .get_column("average")
                    .item()
                )

                percent_diff = (average / all_average * 100) - 100
                if abs(percent_diff) > cls.THRESHOLD:
                    significant_differences.append(
                        {
                            "cluster": cluster,
                            "kingdom": kingdom,
                            "taxonRank": taxonRank,
                            "scientificName": scientificName,
                            "percentage_difference": percent_diff,
                        }
                    )

        # Create a DataFrame from the significant differences
        df = pl.DataFrame(significant_differences, schema=cls.SCHEMA)
        return cls(df)
