import math

import dataframely as dy
import numpy as np
import polars as pl
from scipy.stats import fisher_exact

from src.dataframes.cluster_neighbors import ClusterNeighborsSchema
from src.dataframes.cluster_taxa_statistics import (
    ClusterTaxaStatisticsSchema,
    iter_cluster_ids,
)


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
    high_log2_high_count_score = dy.Float64(nullable=False)
    low_log2_high_count_score = dy.Float64(nullable=False)

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
                        "high_log2_high_count_score": [],
                        "low_log2_high_count_score": [],
                    },
                    schema={
                        "cluster": pl.UInt32,
                        "taxonId": pl.UInt32,
                        "p_value": pl.Float64,
                        "log2_fold_change": pl.Float64,
                        "cluster_count": pl.UInt32,
                        "neighbor_count": pl.UInt32,
                        "high_log2_high_count_score": pl.Float64,
                        "low_log2_high_count_score": pl.Float64,
                    },
                )
            )

        df = pl.DataFrame(significant_differences).with_columns(
            pl.col("cluster").cast(pl.UInt32),
            pl.col("taxonId").cast(pl.UInt32),
            pl.col("cluster_count").cast(pl.UInt32),
            pl.col("neighbor_count").cast(pl.UInt32),
        )

        # Calculate scoring metrics to identify interesting taxa
        # These scores help prioritize which taxa are most characteristic of each cluster

        # Get min/max values for normalization
        # These can be None if the dataframe is empty, so we handle that case
        log2_min = df["log2_fold_change"].min()
        log2_max = df["log2_fold_change"].max()
        cluster_min = df["cluster_count"].min()
        cluster_max = df["cluster_count"].max()

        # Skip scoring if we don't have valid ranges (empty dataframe or all same values)
        if (
            log2_min is None
            or log2_max is None
            or cluster_min is None
            or cluster_max is None
            or log2_min == log2_max
            or cluster_min == cluster_max
        ):
            # If we can't normalize, set all scores to 0
            df = df.with_columns(
                [
                    pl.lit(0.0).alias("high_log2_high_count_score"),
                    pl.lit(0.0).alias("low_log2_high_count_score"),
                ]
            )
        else:
            # Type assertions: after the None checks above, we know these are numeric
            assert isinstance(log2_min, (int, float))
            assert isinstance(log2_max, (int, float))
            assert isinstance(cluster_min, (int, float))
            assert isinstance(cluster_max, (int, float))

            # Step 1: Normalization to scale both metrics to [0, 1] range
            #
            # log2_fold_change: LINEAR normalization (it's already on log scale)
            # - Formula: (value - min) / (max - min)
            #
            # cluster_count: LOGARITHMIC normalization (reduces dominance of top counts)
            # - cluster_count ranges from ~5 to ~700K (massive range!)
            # - Linear normalization would make only top 1% of counts score well
            # - Log normalization: (log(value) - log(min)) / (log(max) - log(min))
            # - This gives more credit to moderately common taxa (1K-100K observations)
            # - Example: count=10K gets log_norm≈0.64 vs linear_norm≈0.01
            df = df.with_columns(
                [
                    # Linear normalization for log2_fold_change
                    (
                        (pl.col("log2_fold_change") - float(log2_min))
                        / (float(log2_max) - float(log2_min))
                    ).alias("log2_norm"),
                    # Logarithmic normalization for cluster_count
                    (
                        (pl.col("cluster_count").log() - math.log(float(cluster_min)))
                        / (math.log(float(cluster_max)) - math.log(float(cluster_min)))
                    ).alias("cluster_norm"),
                ]
            )

            # Step 2: Calculate INDEPENDENT composite scores
            # These scores are now truly independent - they don't sum to a constant
            df = df.with_columns(
                [
                    # high_log2_high_count_score: Product of both normalized values
                    # This identifies taxa that are BOTH highly enriched AND very common
                    # - Score is high only when BOTH metrics are high
                    # - If either metric is low, the product is low (geometric mean behavior)
                    # - Rare but highly enriched taxa score low here (need high count too)
                    # - Common but not enriched taxa score low here (need high log2 too)
                    # Example: log2_norm=0.9, cluster_norm=0.8 → score=0.72 (good balance)
                    #          log2_norm=0.9, cluster_norm=0.1 → score=0.09 (too rare)
                    #          log2_norm=0.1, cluster_norm=0.9 → score=0.09 (not enriched)
                    (pl.col("log2_norm") * pl.col("cluster_norm")).alias(
                        "high_log2_high_count_score"
                    ),
                    # low_log2_high_count_score: Only for taxa with NEGATIVE fold change
                    # This identifies "ubiquitous" taxa that are common but depleted/not enriched
                    # - Only taxa with log2_fold_change < 0 get a non-zero score
                    # - For negative fold changes: score = (1 - log2_norm) × cluster_norm
                    # - For positive/zero fold changes: score = 0
                    # - This makes the scores truly independent - taxa with positive fold change
                    #   cannot score high on low_log2 (they get 0)
                    # Example: log2=-2.0, cluster_norm=0.9 → score might be 0.7 (depleted but common)
                    #          log2=+2.0, cluster_norm=0.9 → score=0.0 (enriched, not depleted)
                    #          log2=-2.0, cluster_norm=0.1 → score might be 0.07 (depleted but rare)
                    #
                    # NOTE: These two scores are now truly independent!
                    # - high_log2_high_count requires positive high fold change + high count
                    # - low_log2_high_count requires negative fold change + high count
                    # - Taxa with middle/positive fold change score 0 on low_log2
                    # - Taxa with negative fold change can score on both if count is high enough
                    (
                        pl.when(pl.col("log2_fold_change") < 0)
                        .then((1.0 - pl.col("log2_norm")) * pl.col("cluster_norm"))
                        .otherwise(0.0)
                    ).alias("low_log2_high_count_score"),
                ]
            )

            # Drop temporary normalization columns - we only need the final scores
            df = df.drop(["log2_norm", "cluster_norm"])

        return cls.validate(df)
