import unittest

import numpy as np
import polars as pl

from src.cluster_optimization import (
    format_optimization_results,
    get_overall_silhouette_scores,
    optimize_num_clusters,
    optimize_num_clusters_multi_metric,
    select_optimal_k,
)
from src.dataframes.geocode_cluster_metrics import (
    GeocodeClusterMetricsSchema,
    _find_elbow_point,
    _compute_inertia,
    get_elbow_analysis,
    select_optimal_k_multi_metric,
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
                "calinski_harabasz_score": [150.0, 200.0, 250.0, 240.0, 220.0, 200.0, 180.0],
                "davies_bouldin_score": [1.5, 1.2, 0.9, 1.0, 1.1, 1.2, 1.3],
                # Classic elbow pattern: steep drop then plateau
                "inertia": [1000.0, 600.0, 350.0, 280.0, 250.0, 230.0, 215.0],
                "silhouette_normalized": [0.675, 0.71, 0.74, 0.725, 0.70, 0.69, 0.675],
                "calinski_harabasz_normalized": [0.0, 0.357, 0.714, 0.643, 0.5, 0.357, 0.214],
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
        # Linear decrease - elbow point should be middle-ish
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

        # For linear, elbow detection should still return something
        elbow_k = _find_elbow_point(linear_df)
        self.assertIsNotNone(elbow_k)
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
        optimal_k = select_optimal_k_multi_metric(
            self.mock_metrics_df,
            selection_method="elbow",
        )

        self.assertIsNotNone(optimal_k)
        # Elbow should be detected in a reasonable range
        self.assertIn(optimal_k, [3, 4, 5])

    def test_select_optimal_k_elbow_ignores_threshold(self):
        """Test that elbow method ignores silhouette threshold."""
        # Set a very high threshold that would filter out all k values
        optimal_k = select_optimal_k_multi_metric(
            self.mock_metrics_df,
            min_silhouette_threshold=0.99,  # Would filter all for other methods
            selection_method="elbow",
        )

        # Elbow method should still return a result
        self.assertIsNotNone(optimal_k)


class TestComputeInertia(unittest.TestCase):
    """Tests for within-cluster sum of squares computation."""

    def test_compute_inertia_single_cluster(self):
        """Test inertia computation for single cluster."""
        dm_square = np.array([
            [0.0, 0.5, 0.6],
            [0.5, 0.0, 0.3],
            [0.6, 0.3, 0.0],
        ])
        labels = np.array([0, 0, 0])  # All in one cluster

        inertia = _compute_inertia(dm_square, labels)
        self.assertGreater(inertia, 0)

    def test_compute_inertia_multiple_clusters(self):
        """Test inertia computation for multiple clusters."""
        dm_square = np.array([
            [0.0, 0.1, 0.8, 0.9],
            [0.1, 0.0, 0.9, 0.8],
            [0.8, 0.9, 0.0, 0.1],
            [0.9, 0.8, 0.1, 0.0],
        ])
        labels = np.array([0, 0, 1, 1])  # Two clusters

        inertia = _compute_inertia(dm_square, labels)
        self.assertGreater(inertia, 0)

    def test_compute_inertia_singleton_clusters(self):
        """Test that singleton clusters contribute zero inertia."""
        dm_square = np.array([
            [0.0, 0.5],
            [0.5, 0.0],
        ])
        labels = np.array([0, 1])  # Each point in its own cluster

        inertia = _compute_inertia(dm_square, labels)
        self.assertEqual(inertia, 0.0)

    def test_compute_inertia_decreases_with_more_clusters(self):
        """Test that inertia generally decreases with more clusters."""
        # Create a simple distance matrix
        dm_square = np.array([
            [0.0, 0.2, 0.8, 0.9],
            [0.2, 0.0, 0.9, 0.8],
            [0.8, 0.9, 0.0, 0.2],
            [0.9, 0.8, 0.2, 0.0],
        ])

        # One cluster (all together)
        inertia_1 = _compute_inertia(dm_square, np.array([0, 0, 0, 0]))

        # Two clusters
        inertia_2 = _compute_inertia(dm_square, np.array([0, 0, 1, 1]))

        # Three clusters
        inertia_3 = _compute_inertia(dm_square, np.array([0, 0, 1, 2]))

        # Inertia should decrease as we add more clusters
        self.assertGreater(inertia_1, inertia_2)
        self.assertGreaterEqual(inertia_2, inertia_3)


class TestMultiMetricSelection(unittest.TestCase):
    """Tests for multi-metric cluster selection methods."""

    def setUp(self):
        """Create mock metrics data for testing selection methods."""
        self.mock_metrics_data = pl.DataFrame(
            {
                "num_clusters": [2, 3, 4, 5, 6],
                "silhouette_score": [0.30, 0.42, 0.45, 0.40, 0.35],
                "calinski_harabasz_score": [100.0, 180.0, 200.0, 160.0, 120.0],
                "davies_bouldin_score": [1.8, 1.2, 0.9, 1.1, 1.4],
                "inertia": [800.0, 400.0, 250.0, 200.0, 180.0],
                "silhouette_normalized": [0.65, 0.71, 0.725, 0.70, 0.675],
                "calinski_harabasz_normalized": [0.0, 0.8, 1.0, 0.6, 0.2],
                "davies_bouldin_normalized": [0.0, 0.667, 1.0, 0.778, 0.444],
                "inertia_normalized": [0.0, 0.645, 0.887, 0.968, 1.0],
                "combined_score": [0.26, 0.73, 0.91, 0.69, 0.44],
            }
        ).with_columns(pl.col("num_clusters").cast(pl.UInt32))
        self.mock_metrics_df = GeocodeClusterMetricsSchema.validate(
            self.mock_metrics_data
        )

    def test_select_combined_method(self):
        """Test selection using combined score."""
        optimal_k = select_optimal_k_multi_metric(
            self.mock_metrics_df,
            selection_method="combined",
        )

        # k=4 has highest combined score (0.91)
        self.assertEqual(optimal_k, 4)

    def test_select_silhouette_method(self):
        """Test selection using silhouette score only."""
        optimal_k = select_optimal_k_multi_metric(
            self.mock_metrics_df,
            selection_method="silhouette",
        )

        # k=4 has highest silhouette (0.45)
        self.assertEqual(optimal_k, 4)

    def test_select_with_silhouette_threshold(self):
        """Test that silhouette threshold filters k values."""
        # k=2 has silhouette 0.30, below threshold of 0.35
        optimal_k = select_optimal_k_multi_metric(
            self.mock_metrics_df,
            selection_method="combined",
            min_silhouette_threshold=0.35,
        )

        # k=2 should be excluded
        self.assertNotEqual(optimal_k, 2)

    def test_select_returns_none_when_all_filtered(self):
        """Test that None is returned when all k values are below threshold."""
        optimal_k = select_optimal_k_multi_metric(
            self.mock_metrics_df,
            selection_method="combined",
            min_silhouette_threshold=0.99,  # No k meets this
        )

        self.assertIsNone(optimal_k)

    def test_invalid_selection_method(self):
        """Test that invalid selection method raises ValueError."""
        with self.assertRaises(ValueError):
            select_optimal_k_multi_metric(
                self.mock_metrics_df,
                selection_method="invalid_method",
            )


class TestMultiMetricIntegration(unittest.TestCase):
    """Integration tests for multi-metric cluster optimization."""

    def test_optimize_multi_metric_elbow(self):
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
        optimal_k, metrics_df = optimize_num_clusters_multi_metric(
            distance_matrix,
            cluster_df,
            selection_method="elbow",
        )

        self.assertIsNotNone(optimal_k)
        self.assertIn(optimal_k, [2, 3, 4])

        # Check that metrics_df has inertia column
        self.assertIn("inertia", metrics_df.columns)

    def test_optimize_multi_metric_all_methods(self):
        """Test that all selection methods work with the optimizer."""
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
            max_k=4,  # Need at least 3 k values for elbow method
        )

        for method in ["combined", "silhouette", "elbow"]:
            optimal_k, metrics_df = optimize_num_clusters_multi_metric(
                distance_matrix,
                cluster_df,
                selection_method=method,
            )

            self.assertIsNotNone(optimal_k, f"Method {method} returned None")
            self.assertIn(optimal_k, [2, 3, 4], f"Method {method} returned invalid k")


if __name__ == "__main__":
    unittest.main()
