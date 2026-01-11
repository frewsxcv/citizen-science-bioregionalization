import unittest

import numpy as np
import polars as pl

from src.cluster_optimization import (
    format_optimization_results,
    get_overall_silhouette_scores,
    optimize_num_clusters,
    select_optimal_k,
)
from src.dataframes.geocode_silhouette_score import GeocodeSilhouetteScoreSchema
from src.matrices.geocode_connectivity import GeocodeConnectivityMatrix
from src.matrices.geocode_distance import GeocodeDistanceMatrix
from test.fixtures.geocode import mock_geocode_no_edges_df
from test.fixtures.geocode_neighbors import mock_geocode_neighbors_df


class TestClusterOptimization(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        # Create mock silhouette score data for multiple k values
        self.mock_silhouette_data = pl.DataFrame(
            {
                "geocode": [None, 1000, 2000, 3000, None, 1000, 2000, 3000],
                "silhouette_score": [0.45, 0.42, 0.48, 0.43, 0.38, 0.35, 0.40, 0.36],
                "num_clusters": [5, 5, 5, 5, 8, 8, 8, 8],
            }
        ).with_columns(
            pl.col("geocode").cast(pl.UInt64, strict=False),
            pl.col("silhouette_score").cast(pl.Float64),
            pl.col("num_clusters").cast(pl.UInt32),
        )
        self.mock_silhouette_df = GeocodeSilhouetteScoreSchema.validate(
            self.mock_silhouette_data
        )

    def test_get_overall_silhouette_scores(self):
        """Test extraction of overall silhouette scores."""
        overall_scores = get_overall_silhouette_scores(self.mock_silhouette_df)

        # Should have 2 rows (one for k=5, one for k=8)
        self.assertEqual(len(overall_scores), 2)

        # Should be sorted by silhouette_score descending
        self.assertEqual(overall_scores["num_clusters"][0], 5)  # Higher score
        self.assertEqual(overall_scores["num_clusters"][1], 8)  # Lower score

        # Check actual scores
        self.assertAlmostEqual(overall_scores["silhouette_score"][0], 0.45, places=2)
        self.assertAlmostEqual(overall_scores["silhouette_score"][1], 0.38, places=2)

    def test_select_optimal_k_with_threshold(self):
        """Test selection of optimal k with threshold."""
        optimal_k = select_optimal_k(self.mock_silhouette_df, min_threshold=0.25)

        # Should select k=5 (highest score above threshold)
        self.assertEqual(optimal_k, 5)

    def test_select_optimal_k_high_threshold(self):
        """Test selection with threshold that filters all results."""
        optimal_k = select_optimal_k(self.mock_silhouette_df, min_threshold=0.90)

        # Should return None (no k meets threshold)
        self.assertIsNone(optimal_k)

    def test_select_optimal_k_no_threshold(self):
        """Test selection without threshold."""
        optimal_k = select_optimal_k(self.mock_silhouette_df, min_threshold=None)

        # Should select k=5 (highest score regardless of threshold)
        self.assertEqual(optimal_k, 5)

    def test_format_optimization_results(self):
        """Test formatting of optimization results."""
        results_table = format_optimization_results(self.mock_silhouette_df)

        # Check structure
        self.assertIn("rank", results_table.columns)
        self.assertIn("num_clusters", results_table.columns)
        self.assertIn("silhouette_score", results_table.columns)

        # Check ordering (should be sorted by score descending)
        self.assertEqual(results_table["rank"][0], 1)
        self.assertEqual(results_table["num_clusters"][0], 5)
        self.assertEqual(results_table["rank"][1], 2)
        self.assertEqual(results_table["num_clusters"][1], 8)

    def test_optimize_num_clusters_basic(self):
        """Test basic functionality of optimize_num_clusters."""
        # Create minimal test data with 5 geocodes
        geocode_lf = mock_geocode_no_edges_df().lazy()

        # Create distance matrix (condensed format for 5 geocodes = 10 distances)
        # For 5 points: (0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)
        condensed_distances = np.array(
            [0.2, 0.8, 0.9, 0.85, 0.15, 0.7, 0.75, 0.8, 0.85, 0.3]
        )
        # Create mock reduced features (5 geocodes x 2 dimensions)
        reduced_features = np.array(
            [[0.1, 0.2], [0.15, 0.25], [0.8, 0.9], [0.85, 0.95], [0.5, 0.5]]
        )
        distance_matrix = GeocodeDistanceMatrix(condensed_distances, reduced_features)

        # Create connectivity matrix from neighbors
        geocode_neighbors_df = mock_geocode_neighbors_df()
        connectivity_matrix = GeocodeConnectivityMatrix.build(geocode_neighbors_df)

        # Build clustering for k=2 to k=3
        from src.dataframes.geocode_cluster import build_geocode_cluster_multi_k_df

        cluster_df = build_geocode_cluster_multi_k_df(
            geocode_lf,
            distance_matrix,
            connectivity_matrix,
            min_k=2,
            max_k=3,
        )

        # Find optimal k
        optimal_k, combined_scores = optimize_num_clusters(
            distance_matrix,
            cluster_df,
        )

        # Check that we got a valid result
        self.assertIsNotNone(optimal_k)
        self.assertIn(optimal_k, [2, 3])

        # Check that combined_scores has the right structure
        self.assertIn("geocode", combined_scores.columns)
        self.assertIn("silhouette_score", combined_scores.columns)
        self.assertIn("num_clusters", combined_scores.columns)

        # Should have overall scores for k=2 and k=3
        overall_scores = get_overall_silhouette_scores(combined_scores)
        self.assertEqual(len(overall_scores), 2)
        self.assertTrue(2 in overall_scores["num_clusters"].to_list())
        self.assertTrue(3 in overall_scores["num_clusters"].to_list())

    def test_optimize_num_clusters_invalid_range(self):
        """Test that build_geocode_cluster_multi_k_df validates k range."""
        from src.dataframes.geocode_cluster import build_geocode_cluster_multi_k_df

        geocode_lf = mock_geocode_no_edges_df().lazy()

        # Create condensed distance matrix for 5 geocodes = 10 distances
        condensed_distances = np.array(
            [0.2, 0.8, 0.9, 0.85, 0.15, 0.7, 0.75, 0.8, 0.85, 0.3]
        )
        # Create mock reduced features (5 geocodes x 2 dimensions)
        reduced_features = np.array(
            [[0.1, 0.2], [0.15, 0.25], [0.8, 0.9], [0.85, 0.95], [0.5, 0.5]]
        )
        distance_matrix = GeocodeDistanceMatrix(condensed_distances, reduced_features)
        geocode_neighbors_df = mock_geocode_neighbors_df()
        connectivity_matrix = GeocodeConnectivityMatrix.build(geocode_neighbors_df)

        # Test min_k < 2
        with self.assertRaises(ValueError):
            build_geocode_cluster_multi_k_df(
                geocode_lf,
                distance_matrix,
                connectivity_matrix,
                min_k=1,
                max_k=5,
            )

        # Test max_k < min_k
        with self.assertRaises(ValueError):
            build_geocode_cluster_multi_k_df(
                geocode_lf,
                distance_matrix,
                connectivity_matrix,
                min_k=5,
                max_k=3,
            )

    def test_select_optimal_k_with_negative_scores(self):
        """Test selection when some scores are negative."""
        # Create data with negative scores
        negative_score_data = pl.DataFrame(
            {
                "geocode": [None, 1000, 2000, None, 1000, 2000],
                "silhouette_score": [0.35, 0.32, 0.38, -0.15, -0.12, -0.18],
                "num_clusters": [3, 3, 3, 10, 10, 10],
            }
        ).with_columns(
            pl.col("geocode").cast(pl.UInt64, strict=False),
            pl.col("silhouette_score").cast(pl.Float64),
            pl.col("num_clusters").cast(pl.UInt32),
        )
        negative_score_df = GeocodeSilhouetteScoreSchema.validate(negative_score_data)

        # With threshold, should select k=3 (only positive score above threshold)
        optimal_k = select_optimal_k(negative_score_df, min_threshold=0.25)
        self.assertEqual(optimal_k, 3)

        # Without threshold, should still select k=3 (highest score)
        optimal_k_no_threshold = select_optimal_k(negative_score_df, min_threshold=None)
        self.assertEqual(optimal_k_no_threshold, 3)


if __name__ == "__main__":
    unittest.main()
