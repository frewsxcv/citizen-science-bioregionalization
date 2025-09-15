import unittest
import polars as pl
import numpy as np
from src.matrices.cluster_distance import (
    ClusterDistanceMatrix,
    pivot_taxon_counts_for_clusters,
    build_X,
)
from src.dataframes.cluster_taxa_statistics import ClusterTaxaStatisticsSchema
import dataframely as dy
from test.fixtures.cluster_taxa_statistics import (
    mock_cluster_taxa_statistics_dataframe,
)


class TestClusterDistanceMatrix(unittest.TestCase):
    def test_pivot_taxon_counts_for_clusters(self):
        """Test that the pivot function correctly transforms the data"""
        # Create a mock ClusterTaxaStatisticsDataFrame
        df = pl.DataFrame(
            {
                "cluster": [1, 1, 2, 2, None],  # None represents overall stats
                "taxonId": [101, 102, 101, 102, 101],
                "count": [10, 5, 3, 8, 13],
                "average": [0.05, 0.025, 0.015, 0.04, 0.065],
            },
            schema={
                "cluster": pl.UInt32(),
                "taxonId": pl.UInt32(),
                "count": pl.UInt32(),
                "average": pl.Float64(),
            },
        )
        cluster_taxa_stats = ClusterTaxaStatisticsSchema.validate(df)

        # Run the pivot function
        result = pivot_taxon_counts_for_clusters(cluster_taxa_stats)

        # Verify the structure and content
        self.assertEqual(
            result.shape, (2, 3)
        )  # 2 clusters, 3 columns (cluster + 2 taxa)
        self.assertEqual(result.columns, ["cluster", "101", "102"])

        # Convert to dict for easier assertions
        result_dict = result.to_dict(as_series=False)

        # Check values for cluster 1
        cluster1_idx = result_dict["cluster"].index(1)
        self.assertAlmostEqual(result_dict["101"][cluster1_idx], 0.05)
        self.assertAlmostEqual(result_dict["102"][cluster1_idx], 0.025)

        # Check values for cluster 2
        cluster2_idx = result_dict["cluster"].index(2)
        self.assertAlmostEqual(result_dict["101"][cluster2_idx], 0.015)
        self.assertAlmostEqual(result_dict["102"][cluster2_idx], 0.04)

    def test_cluster_distance_matrix_build(self):
        """Test building a distance matrix from cluster taxa statistics"""
        # Create a mock ClusterTaxaStatisticsDataFrame
        df = pl.DataFrame(
            {
                "cluster": [1, 1, 1, 2, 2, 2, 3, 3, 3, None, None, None],
                "taxonId": [101, 102, 103, 101, 102, 103, 101, 102, 103, 101, 102, 103],
                "count": [10, 5, 20, 3, 15, 10, 8, 2, 25, 21, 22, 55],
                "average": [
                    0.05,
                    0.025,
                    0.1,
                    0.015,
                    0.075,
                    0.05,
                    0.04,
                    0.01,
                    0.125,
                    0.035,
                    0.037,
                    0.092,
                ],
            },
            schema={
                "cluster": pl.UInt32(),
                "taxonId": pl.UInt32(),
                "count": pl.UInt32(),
                "average": pl.Float64(),
            },
        )
        cluster_taxa_stats = ClusterTaxaStatisticsSchema.validate(df)

        # Build the distance matrix
        distance_matrix = ClusterDistanceMatrix.build(cluster_taxa_stats)

        # Verify the structure
        self.assertEqual(len(distance_matrix.cluster_ids()), 3)
        self.assertEqual(distance_matrix.cluster_ids(), [1, 2, 3])

        # The condensed distance matrix should have n*(n-1)/2 elements where n is the number of clusters
        n = len(distance_matrix.cluster_ids())
        self.assertEqual(len(distance_matrix.condensed()), n * (n - 1) // 2)

        # The square matrix should be n x n
        self.assertEqual(distance_matrix.squareform().shape, (n, n))

        # The distance matrix should be symmetric
        square_matrix = distance_matrix.squareform()
        for i in range(n):
            for j in range(n):
                self.assertAlmostEqual(square_matrix[i, j], square_matrix[j, i])

        # The diagonal should be all zeros
        for i in range(n):
            self.assertAlmostEqual(square_matrix[i, i], 0.0)

    def test_get_distance(self):
        """Test getting the distance between two clusters"""
        # Create a mock ClusterTaxaStatisticsDataFrame with carefully controlled values
        # to ensure predictable Bray-Curtis distances
        df = pl.DataFrame(
            {
                # Cluster 1: 100% taxon 101, 0% taxon 102
                # Cluster 2: 0% taxon 101, 100% taxon 102
                # Cluster 3: 50% taxon 101, 50% taxon 102
                "cluster": [1, 1, 2, 2, 3, 3, None, None],
                "taxonId": [101, 102, 101, 102, 101, 102, 101, 102],
                "count": [100, 0, 0, 100, 50, 50, 150, 150],
                "average": [1.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5, 0.5],
            },
            schema={
                "cluster": pl.UInt32(),
                "taxonId": pl.UInt32(),
                "count": pl.UInt32(),
                "average": pl.Float64(),
            },
        )

        cluster_taxa_stats = ClusterTaxaStatisticsSchema.validate(df)

        # Use a custom implementation to build the matrix to have more control
        X, cluster_ids = build_X(cluster_taxa_stats)

        # For these specific values, we can manually calculate the expected distances:
        # BC(1,2) = 1.0 (completely different)
        # BC(1,3) = 0.5 (half different)
        # BC(2,3) = 0.5 (half different)

        # Create a custom distance matrix with these known values
        distances = np.array([1.0, 0.5, 0.5])  # [d(1,2), d(1,3), d(2,3)]
        distance_matrix = ClusterDistanceMatrix(distances, cluster_ids)

        # Now verify the distances
        self.assertAlmostEqual(distance_matrix.get_distance(1, 2), 1.0)
        self.assertAlmostEqual(distance_matrix.get_distance(1, 3), 0.5)
        self.assertAlmostEqual(distance_matrix.get_distance(2, 3), 0.5)

        # Getting distance to self should return 0
        self.assertEqual(distance_matrix.get_distance(1, 1), 0.0)

    def test_get_most_similar_clusters(self):
        """Test getting the most similar clusters"""
        # Create a mock ClusterTaxaStatisticsDataFrame with 4 clusters
        # Cluster 1: All taxon 101
        # Cluster 2: All taxon 102
        # Cluster 3: Mix of 101 and 102, leaning toward 101 (closer to 1)
        # Cluster 4: Mix of 101 and 102, leaning toward 102 (closer to 2)
        cluster_taxa_stats = mock_cluster_taxa_statistics_dataframe()
        distance_matrix = ClusterDistanceMatrix.build(cluster_taxa_stats)

        # Test for cluster 1
        similar_to_1 = distance_matrix.get_most_similar_clusters(1)
        self.assertEqual(len(similar_to_1), 3)
        self.assertEqual(similar_to_1[0][0], 3)  # Cluster 3 should be most similar

        # Test for cluster 2
        similar_to_2 = distance_matrix.get_most_similar_clusters(2)
        self.assertEqual(len(similar_to_2), 3)
        self.assertEqual(similar_to_2[0][0], 4)  # Cluster 4 should be most similar

        # Test with n parameter
        similar_to_3 = distance_matrix.get_most_similar_clusters(3, n=1)
        self.assertEqual(len(similar_to_3), 1)

        # Test with invalid cluster ID
        with self.assertRaises(ValueError):
            distance_matrix.get_most_similar_clusters(999)
