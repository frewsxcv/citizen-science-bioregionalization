import polars as pl
import dataframely as dy
from src.dataframes.cluster_neighbors import ClusterNeighborsSchema
from src.dataframes.cluster_taxa_statistics import (
    ClusterTaxaStatisticsSchema,
    iter_cluster_ids,
)
from scipy.stats import fisher_exact
import numpy as np


class ClusterSignificantDifferencesSchema(dy.Schema):
    """
    A dataframe that contains the significant differences between clusters.
    """

    P_VALUE_THRESHOLD = 0.05
    MIN_COUNT_THRESHOLD = 5

    cluster = dy.UInt32(nullable=False)
    taxonId = dy.UInt32(nullable=False)
    p_value = dy.Float64(nullable=False)
    log2_fold_change = dy.Float64(nullable=False)
    cluster_count = dy.UInt32(nullable=False)
    neighbor_count = dy.UInt32(nullable=False)

    @classmethod
    def build(
        cls,
        all_stats: dy.DataFrame[ClusterTaxaStatisticsSchema],
        cluster_neighbors: dy.DataFrame[ClusterNeighborsSchema],
    ) -> dy.DataFrame["ClusterSignificantDifferencesSchema"]:
        significant_differences = []

        neighbors_map = {
            row["cluster"]: row["direct_and_indirect_neighbors"]
            for row in cluster_neighbors.iter_rows(named=True)
        }

        for cluster in iter_cluster_ids(all_stats):
            if cluster is None:
                continue

            neighbors = neighbors_map.get(cluster)
            if not neighbors:
                continue

            cluster_stats = all_stats.filter(pl.col("cluster") == cluster)
            neighbor_stats = all_stats.filter(pl.col("cluster").is_in(neighbors))

            total_cluster_count = cluster_stats.get_column("count").sum()
            total_neighbor_count = neighbor_stats.get_column("count").sum()

            for taxonId, count in (
                cluster_stats.sort(by="count", descending=True)
                .select(["taxonId", "count"])
                .iter_rows(named=False)
            ):
                if count < cls.MIN_COUNT_THRESHOLD:
                    continue

                neighbor_count = (
                    neighbor_stats.filter(pl.col("taxonId") == taxonId)
                    .get_column("count")
                    .sum()
                )

                if neighbor_count < cls.MIN_COUNT_THRESHOLD:
                    continue

                # Create contingency table
                table = np.array(
                    [
                        [count, total_cluster_count - count],
                        [neighbor_count, total_neighbor_count - neighbor_count],
                    ]
                )

                # Perform Fisher's exact test
                odds_ratio, p_value = fisher_exact(table, alternative="two-sided")

                if p_value < cls.P_VALUE_THRESHOLD:
                    # Calculate log2 fold change
                    mean_cluster = count / total_cluster_count
                    mean_neighbor = neighbor_count / total_neighbor_count

                    if mean_cluster == 0 or mean_neighbor == 0:
                        continue

                    log2_fold_change = np.log2(mean_cluster / mean_neighbor)

                    significant_differences.append(
                        {
                            "cluster": cluster,
                            "taxonId": taxonId,
                            "p_value": p_value,
                            "log2_fold_change": log2_fold_change,
                            "cluster_count": count,
                            "neighbor_count": neighbor_count,
                        }
                    )

        if not significant_differences:
            return cls.validate(
                pl.DataFrame(
                    {
                        "cluster": [],
                        "taxonId": [],
                        "p_value": [],
                        "log2_fold_change": [],
                        "cluster_count": [],
                        "neighbor_count": [],
                    },
                    schema={
                        "cluster": pl.UInt32,
                        "taxonId": pl.UInt32,
                        "p_value": pl.Float64,
                        "log2_fold_change": pl.Float64,
                        "cluster_count": pl.UInt32,
                        "neighbor_count": pl.UInt32,
                    },
                )
            )

        df = pl.DataFrame(significant_differences).with_columns(
            pl.col("cluster").cast(pl.UInt32),
            pl.col("taxonId").cast(pl.UInt32),
            pl.col("cluster_count").cast(pl.UInt32),
            pl.col("neighbor_count").cast(pl.UInt32),
        )
        return cls.validate(df)
