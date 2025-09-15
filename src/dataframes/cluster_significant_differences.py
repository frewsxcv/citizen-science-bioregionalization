import polars as pl
import dataframely as dy
from src.dataframes.cluster_taxa_statistics import ClusterTaxaStatisticsDataFrame


class ClusterSignificantDifferencesSchema(dy.Schema):
    """
    A dataframe that contains the significant differences between clusters.
    """

    THRESHOLD = 10  # Percent difference

    cluster = dy.UInt32(nullable=False)
    taxonId = dy.UInt32(nullable=False)
    percentage_difference = dy.Float64(
        nullable=False
    )  # TODO: should this a p-value?

    @classmethod
    def build(
        cls, all_stats: ClusterTaxaStatisticsDataFrame
    ) -> dy.DataFrame["ClusterSignificantDifferencesSchema"]:
        # Calculate significant differences
        significant_differences = []

        for cluster in all_stats.iter_cluster_ids():
            for taxonId, average in (
                all_stats.df.filter(
                    (
                        pl.col("cluster").is_null()
                        if cluster is None
                        else pl.col("cluster") == cluster
                    ),
                )
                .sort(by="count", descending=True)
                .limit(20)  # TODO: Does this need to happen?
                .select(["taxonId", "average"])
                .iter_rows(named=False)
            ):
                all_average = (
                    all_stats.df.filter(
                        pl.col("taxonId") == taxonId,
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
                            "taxonId": taxonId,
                            "percentage_difference": percent_diff,
                        }
                    )

        # Create a DataFrame from the significant differences
        df = pl.DataFrame(significant_differences).with_columns(
            pl.col("cluster").cast(pl.UInt32),
            pl.col("taxonId").cast(pl.UInt32),
        )
        return cls.validate(df)
