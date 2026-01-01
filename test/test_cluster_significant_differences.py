import unittest

import dataframely as dy
import polars as pl

from src.dataframes.cluster_neighbors import ClusterNeighborsSchema
from src.dataframes.cluster_significant_differences import (
    ClusterSignificantDifferencesSchema,
    build_cluster_significant_differences_df,
)
from src.dataframes.cluster_taxa_statistics import ClusterTaxaStatisticsSchema


class TestClusterSignificantDifferences(unittest.TestCase):
    def test_build_significant_differences(self):
        # Mock data to test all filtering conditions with Fisher's exact test
        taxa_stats_df = pl.DataFrame(
            {
                "cluster": [1, 1, 1, 2, 2, 2, None, None, None],
                "taxonId": [101, 102, 103, 101, 102, 103, 101, 102, 103],
                # Taxon 102 will be significant, 101/103 will be filtered by count
                "count": [50, 20, 4, 4, 5, 50, 54, 25, 54],
                "average": [0.8, 0.3, 0.05, 0.05, 0.1, 0.8, 0.4, 0.2, 0.4],
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
        significant_differences_df = build_cluster_significant_differences_df(
            cluster_taxa_stats, cluster_neighbors.lazy()
        )

        # Verify the filtering:
        # - Taxon 101 (cluster 1): cluster_count=50, neighbor_count=4 -> FAIL (neighbor)
        # - Taxon 102 (cluster 1): cluster_count=20, neighbor_count=5 -> PASS
        # - Taxon 103 (cluster 1): cluster_count=4, neighbor_count=50 -> FAIL (cluster)
        # The same logic applies to cluster 2, so we expect 2 rows in the output.
        self.assertEqual(len(significant_differences_df), 2)
        self.assertTrue(
            102 in significant_differences_df.get_column("taxonId").to_list()
        )
        self.assertFalse(
            101 in significant_differences_df.get_column("taxonId").to_list()
        )
        self.assertFalse(
            103 in significant_differences_df.get_column("taxonId").to_list()
        )
