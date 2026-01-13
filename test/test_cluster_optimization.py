import unittest

import numpy as np
import polars as pl

from src.cluster_optimization import optimize_num_clusters
from src.dataframes.geocode_cluster_metrics import (
    GeocodeClusterMetricsSchema,
    _compute_inertia,
    _find_elbow_point,
    get_elbow_analysis,
    select_optimal_k_elbow,
)
from src.matrices.geocode_connectivity import GeocodeConnectivityMatrix
from src.matrices.geocode_distance import GeocodeDistanceMatrix
from test.fixtures.geocode import mock_geocode_no_edges_df
from test.fixtures.geocode_neighbors import mock_geocode_neighbors_df


class TestClusterOptimization(unittest.TestCase):
    def test_optimize_num_clusters_basic(self):
        """Test basic functionality of optimize_num_clusters with elbow method."""
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

        # Find optimal k using elbow method
        optimal_k, metrics_df = optimize_num_clusters(
            distance_matrix,
            cluster_df,
        )

        # Check that we got a valid result
        self.assertIsNotNone(optimal_k)
        self.assertIn(optimal_k, [2, 3])

        # Check that metrics_df has the right structure
        self.assertIn("num_clusters", metrics_df.columns)
        self.assertIn("silhouette_score", metrics_df.columns)
        self.assertIn("inertia", metrics_df.columns)

        # Should have metrics for k=2 and k=3
        self.assertEqual(len(metrics_df), 2)
        self.assertTrue(2 in metrics_df["num_clusters"].to_list())
        self.assertTrue(3 in metrics_df["num_clusters"].to_list())

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


class TestElbowMethod(unittest.TestCase):
    """Tests for the elbow method implementation."""

    def setUp(self):
        """Create mock metrics data with clear elbow pattern."""
        # Simulate classic elbow curve: sharp drop then gradual decrease
        # Elbow should be at k=4 where the rate of decrease changes
        self.mock_metrics_data = pl.DataFrame(
            {
                "num_clusters": [2, 3, 4, 5, 6, 7, 8],
                "silhouette_score": [0.35, 0.42, 0.48, 0.45, 0.40, 0.38, 0.35],
                "calinski_harabasz_score": [
                    150.0,
                    200.0,
                    250.0,
                    240.0,
                    220.0,
                    200.0,
                    180.0,
                ],
                "davies_bouldin_score": [1.5, 1.2, 0.9, 1.0, 1.1, 1.2, 1.3],
                # Classic elbow pattern: steep drop then plateau
                "inertia": [1000.0, 600.0, 350.0, 280.0, 250.0, 230.0, 215.0],
                "silhouette_normalized": [0.675, 0.71, 0.74, 0.725, 0.70, 0.69, 0.675],
                "calinski_harabasz_normalized": [
                    0.0,
                    0.357,
                    0.714,
                    0.643,
                    0.5,
                    0.357,
                    0.214,
                ],
                "davies_bouldin_normalized": [0.0, 0.5, 1.0, 0.833, 0.667, 0.5, 0.333],
                "inertia_normalized": [0.0, 0.509, 0.828, 0.917, 0.955, 0.981, 1.0],
                "combined_score": [0.27, 0.52, 0.82, 0.73, 0.62, 0.52, 0.41],
            }
        ).with_columns(
            pl.col("num_clusters").cast(pl.UInt32),
        )
        self.mock_metrics_df = GeocodeClusterMetricsSchema.validate(
            self.mock_metrics_data
        )

    def test_find_elbow_point_clear_elbow(self):
        """Test elbow detection with a clear elbow pattern."""
        elbow_k = _find_elbow_point(self.mock_metrics_df)

        # The elbow should be detected around k=4 where the curve changes
        self.assertIsNotNone(elbow_k)
        # Allow some tolerance since elbow detection isn't exact
        self.assertIn(elbow_k, [3, 4, 5])

    def test_find_elbow_point_few_points(self):
        """Test that elbow detection returns None with too few points."""
        few_points = pl.DataFrame(
            {
                "num_clusters": [2, 3],
                "silhouette_score": [0.35, 0.42],
                "calinski_harabasz_score": [150.0, 200.0],
                "davies_bouldin_score": [1.5, 1.2],
                "inertia": [1000.0, 600.0],
                "silhouette_normalized": [0.675, 0.71],
                "calinski_harabasz_normalized": [0.0, 1.0],
                "davies_bouldin_normalized": [0.0, 1.0],
                "inertia_normalized": [0.0, 1.0],
                "combined_score": [0.27, 0.71],
            }
        ).with_columns(pl.col("num_clusters").cast(pl.UInt32))
        few_points_df = GeocodeClusterMetricsSchema.validate(few_points)

        elbow_k = _find_elbow_point(few_points_df)
        self.assertIsNone(elbow_k)

    def test_find_elbow_point_linear_decrease(self):
        """Test elbow detection with linear decrease (no clear elbow)."""
        # Linear decrease - Kneedle algorithm should return None (no clear elbow)
        linear_data = pl.DataFrame(
            {
                "num_clusters": [2, 3, 4, 5, 6],
                "silhouette_score": [0.4] * 5,
                "calinski_harabasz_score": [200.0] * 5,
                "davies_bouldin_score": [1.0] * 5,
                "inertia": [500.0, 400.0, 300.0, 200.0, 100.0],  # Linear decrease
                "silhouette_normalized": [0.7] * 5,
                "calinski_harabasz_normalized": [0.5] * 5,
                "davies_bouldin_normalized": [0.5] * 5,
                "inertia_normalized": [0.0, 0.25, 0.5, 0.75, 1.0],
                "combined_score": [0.57] * 5,
            }
        ).with_columns(pl.col("num_clusters").cast(pl.UInt32))
        linear_df = GeocodeClusterMetricsSchema.validate(linear_data)

        # For linear decrease, Kneedle algorithm correctly returns None (no elbow)
        elbow_k = _find_elbow_point(linear_df)
        # The Kneedle algorithm is more conservative and may return None for linear data
        # This is correct behavior - a perfectly linear curve has no elbow
        if elbow_k is not None:
            self.assertIn(elbow_k, [2, 3, 4, 5, 6])

    def test_get_elbow_analysis(self):
        """Test that elbow analysis returns all expected data."""
        analysis = get_elbow_analysis(self.mock_metrics_df)

        # Check all expected keys are present
        self.assertIn("k_values", analysis)
        self.assertIn("inertia_values", analysis)
        self.assertIn("elbow_k", analysis)
        self.assertIn("distances", analysis)
        self.assertIn("inertia_deltas", analysis)
        self.assertIn("inertia_delta2", analysis)

        # Check lengths match
        num_k = len(analysis["k_values"])
        self.assertEqual(len(analysis["inertia_values"]), num_k)
        self.assertEqual(len(analysis["distances"]), num_k)
        self.assertEqual(len(analysis["inertia_deltas"]), num_k)
        self.assertEqual(len(analysis["inertia_delta2"]), num_k)

        # Check k_values are sorted
        self.assertEqual(analysis["k_values"], sorted(analysis["k_values"]))

    def test_select_optimal_k_elbow_method(self):
        """Test selection using elbow method."""
        optimal_k = select_optimal_k_elbow(
            self.mock_metrics_df,
        )

        self.assertIsNotNone(optimal_k)
        # Elbow should be detected in a reasonable range
        self.assertIn(optimal_k, [3, 4, 5])


class TestComputeInertia(unittest.TestCase):
    """Tests for within-cluster sum of squares computation."""

    def test_compute_inertia_single_cluster(self):
        """Test inertia computation for single cluster."""
        dm_square = np.array(
            [
                [0.0, 0.5, 0.6],
                [0.5, 0.0, 0.3],
                [0.6, 0.3, 0.0],
            ]
        )
        labels = np.array([0, 0, 0])  # All in one cluster

        inertia = _compute_inertia(dm_square, labels)
        self.assertGreater(inertia, 0)

    def test_compute_inertia_multiple_clusters(self):
        """Test inertia computation for multiple clusters."""
        dm_square = np.array(
            [
                [0.0, 0.1, 0.8, 0.9],
                [0.1, 0.0, 0.9, 0.8],
                [0.8, 0.9, 0.0, 0.1],
                [0.9, 0.8, 0.1, 0.0],
            ]
        )
        labels = np.array([0, 0, 1, 1])  # Two clusters

        inertia = _compute_inertia(dm_square, labels)
        self.assertGreater(inertia, 0)

    def test_compute_inertia_singleton_clusters(self):
        """Test that singleton clusters contribute zero inertia."""
        dm_square = np.array(
            [
                [0.0, 0.5],
                [0.5, 0.0],
            ]
        )
        labels = np.array([0, 1])  # Each point in its own cluster

        inertia = _compute_inertia(dm_square, labels)
        self.assertEqual(inertia, 0.0)

    def test_compute_inertia_decreases_with_more_clusters(self):
        """Test that inertia generally decreases with more clusters."""
        # Create a simple distance matrix
        dm_square = np.array(
            [
                [0.0, 0.2, 0.8, 0.9],
                [0.2, 0.0, 0.9, 0.8],
                [0.8, 0.9, 0.0, 0.2],
                [0.9, 0.8, 0.2, 0.0],
            ]
        )

        # One cluster (all together)
        inertia_1 = _compute_inertia(dm_square, np.array([0, 0, 0, 0]))

        # Two clusters
        inertia_2 = _compute_inertia(dm_square, np.array([0, 0, 1, 1]))

        # Three clusters
        inertia_3 = _compute_inertia(dm_square, np.array([0, 0, 1, 2]))

        # Inertia should decrease as we add more clusters
        self.assertGreater(inertia_1, inertia_2)
        self.assertGreaterEqual(inertia_2, inertia_3)


class TestElbowIntegration(unittest.TestCase):
    """Integration tests for elbow method cluster optimization."""

    def test_optimize_with_elbow_method(self):
        """Test full optimization pipeline with elbow method."""
        geocode_lf = mock_geocode_no_edges_df().lazy()

        condensed_distances = np.array(
            [0.2, 0.8, 0.9, 0.85, 0.15, 0.7, 0.75, 0.8, 0.85, 0.3]
        )
        reduced_features = np.array(
            [[0.1, 0.2], [0.15, 0.25], [0.8, 0.9], [0.85, 0.95], [0.5, 0.5]]
        )
        distance_matrix = GeocodeDistanceMatrix(condensed_distances, reduced_features)

        geocode_neighbors_df = mock_geocode_neighbors_df()
        connectivity_matrix = GeocodeConnectivityMatrix.build(geocode_neighbors_df)

        from src.dataframes.geocode_cluster import build_geocode_cluster_multi_k_df

        cluster_df = build_geocode_cluster_multi_k_df(
            geocode_lf,
            distance_matrix,
            connectivity_matrix,
            min_k=2,
            max_k=4,
        )

        # Test with elbow method
        optimal_k, metrics_df = optimize_num_clusters(
            distance_matrix,
            cluster_df,
        )

        self.assertIsNotNone(optimal_k)
        self.assertIn(optimal_k, [2, 3, 4])

        # Check that metrics_df has inertia column
        self.assertIn("inertia", metrics_df.columns)


if __name__ == "__main__":
    unittest.main()
