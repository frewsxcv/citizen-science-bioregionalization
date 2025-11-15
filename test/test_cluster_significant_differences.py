import unittest
import polars as pl
import dataframely as dy
from src.dataframes.cluster_significant_differences import (
    ClusterSignificantDifferencesSchema,
)
from src.dataframes.cluster_neighbors import ClusterNeighborsSchema
from src.dataframes.cluster_taxa_statistics import ClusterTaxaStatisticsSchema


class TestClusterSignificantDifferences(unittest.TestCase):
    def test_build_significant_differences(self):
        # Mock cluster taxa statistics with a clear case for significance
        taxa_stats_df = pl.DataFrame(
            {
                "cluster": [1, 1, 2, 2, None, None],
                "taxonId": [101, 102, 101, 102, 101, 102],
                "count": [50, 5, 5, 50, 55, 55],
                "average": [0.9, 0.1, 0.1, 0.9, 0.5, 0.5],
            },
            schema={
                "cluster": pl.UInt32(),
                "taxonId": pl.UInt32(),
                "count": pl.UInt32(),
                "average": pl.Float64(),
            },
        )
        cluster_taxa_stats = ClusterTaxaStatisticsSchema.validate(taxa_stats_df)

        # Mock cluster neighbors
        neighbors_df = pl.DataFrame(
            {
                "cluster": [1, 2],
                "direct_neighbors": [[2], [1]],
                "direct_and_indirect_neighbors": [[2], [1]],
            },
            schema={
                "cluster": pl.UInt32(),
                "direct_neighbors": pl.List(pl.UInt32()),
                "direct_and_indirect_neighbors": pl.List(pl.UInt32()),
            },
        )
        cluster_neighbors = ClusterNeighborsSchema.validate(neighbors_df)

        # Build significant differences
        significant_differences_df = ClusterSignificantDifferencesSchema.build(
            cluster_taxa_stats, cluster_neighbors
        )

        # Verify the results
        self.assertGreater(len(significant_differences_df), 0)

        # Check the case for cluster 1, taxon 101
        diff_c1_t101 = significant_differences_df.filter(
            (pl.col("cluster") == 1) & (pl.col("taxonId") == 101)
        )
        self.assertLess(
            diff_c1_t101.get_column("p_value").item(),
            ClusterSignificantDifferencesSchema.P_VALUE_THRESHOLD,
        )
        self.assertAlmostEqual(
            diff_c1_t101.get_column("log2_fold_change").item(),
            3.3219,
            places=4,
        )