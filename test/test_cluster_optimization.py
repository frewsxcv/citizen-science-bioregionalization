import unittest
from unittest.mock import MagicMock, patch

import dataframely as dy
import numpy as np
import polars as pl
from scipy.sparse import csr_matrix

from src.cluster_optimization import (
    ClusterMetrics,
    ClusterOptimizer,
    OptimalKResult,
    create_metrics_report,
    metrics_to_dataframe,
)
from src.dataframes.geocode import GeocodeNoEdgesSchema
from src.matrices.geocode_connectivity import GeocodeConnectivityMatrix
from src.matrices.geocode_distance import GeocodeDistanceMatrix


class TestClusterOptimization(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        # Create a small mock dataset with 20 geocodes
        n_geocodes = 20

        # Mock geocode dataframe
        self.geocode_dataframe = pl.DataFrame(
            {
                "geocode": list(range(n_geocodes)),
                "center": [b""] * n_geocodes,
                "boundary": [b""] * n_geocodes,
                "direct_neighbors": [[]] * n_geocodes,
                "direct_and_indirect_neighbors": [[]] * n_geocodes,
                "is_edge": [False] * n_geocodes,
            }
        )

        # Mock distance matrix - create synthetic distances
        # Use euclidean distances from random points
        np.random.seed(42)
        points = np.random.rand(n_geocodes, 2)
        square_distances = np.zeros((n_geocodes, n_geocodes))
        for i in range(n_geocodes):
            for j in range(i + 1, n_geocodes):
                dist = np.linalg.norm(points[i] - points[j])
                square_distances[i, j] = dist
                square_distances[j, i] = dist

        # Convert to condensed form
        from scipy.spatial.distance import squareform

        condensed = squareform(square_distances)
        self.distance_matrix = GeocodeDistanceMatrix(condensed)

        # Mock connectivity matrix - create a connected graph
        connectivity = np.zeros((n_geocodes, n_geocodes))
        # Make a linear chain for simplicity
        for i in range(n_geocodes - 1):
            connectivity[i, i + 1] = 1
            connectivity[i + 1, i] = 1

        self.connectivity_matrix = GeocodeConnectivityMatrix(connectivity)

    def test_cluster_metrics_creation(self):
        """Test ClusterMetrics dataclass"""
        metrics = ClusterMetrics(
            k=5,
            silhouette=0.5,
            davies_bouldin=0.8,
            calinski_harabasz=100.0,
            inertia=50.0,
            mean_cluster_size=4,
            min_cluster_size=2,
            max_cluster_size=6,
        )

        self.assertEqual(metrics.k, 5)
        self.assertEqual(metrics.silhouette, 0.5)
        self.assertEqual(metrics.mean_cluster_size, 4)

    def test_optimal_k_result_creation(self):
        """Test OptimalKResult dataclass"""
        metrics = ClusterMetrics(
            k=5,
            silhouette=0.5,
            davies_bouldin=0.8,
            calinski_harabasz=100.0,
            inertia=50.0,
            mean_cluster_size=4,
            min_cluster_size=2,
            max_cluster_size=6,
        )

        result = OptimalKResult(
            optimal_k=5,
            all_metrics=[metrics],
            selection_method="test",
            reason="test reason",
        )

        self.assertEqual(result.optimal_k, 5)
        self.assertEqual(result.selection_method, "test")
        self.assertEqual(len(result.all_metrics), 1)

    def test_optimizer_initialization(self):
        """Test ClusterOptimizer initialization"""
        optimizer = ClusterOptimizer(
            geocode_dataframe=self.geocode_dataframe,
            distance_matrix=self.distance_matrix,
            connectivity_matrix=self.connectivity_matrix,
        )

        self.assertIsNotNone(optimizer)
        self.assertEqual(len(optimizer.geocode_dataframe), 20)

    def test_evaluate_k_range_basic(self):
        """Test evaluating a small range of k values"""
        optimizer = ClusterOptimizer(
            geocode_dataframe=self.geocode_dataframe,
            distance_matrix=self.distance_matrix,
            connectivity_matrix=self.connectivity_matrix,
        )

        # Evaluate k from 2 to 5
        metrics_list = optimizer.evaluate_k_range(k_min=2, k_max=5)

        self.assertEqual(len(metrics_list), 4)  # k=2,3,4,5
        self.assertEqual(metrics_list[0].k, 2)
        self.assertEqual(metrics_list[-1].k, 5)

        # Check that all metrics are calculated
        for metrics in metrics_list:
            self.assertIsInstance(metrics.silhouette, float)
            self.assertIsInstance(metrics.davies_bouldin, float)
            self.assertIsInstance(metrics.calinski_harabasz, float)
            self.assertIsInstance(metrics.inertia, float)
            self.assertGreater(metrics.mean_cluster_size, 0)
            self.assertGreater(metrics.min_cluster_size, 0)

    def test_evaluate_k_range_validates_inputs(self):
        """Test that k_min and k_max are validated"""
        optimizer = ClusterOptimizer(
            geocode_dataframe=self.geocode_dataframe,
            distance_matrix=self.distance_matrix,
            connectivity_matrix=self.connectivity_matrix,
        )

        # k_max too large (>= n_geocodes)
        metrics_list = optimizer.evaluate_k_range(k_min=2, k_max=25)
        # Should be capped at 19 (n_geocodes - 1)
        self.assertLessEqual(max(m.k for m in metrics_list), 19)

        # k_min too small
        metrics_list = optimizer.evaluate_k_range(k_min=1, k_max=5)
        # Should start at 2
        self.assertEqual(min(m.k for m in metrics_list), 2)

    def test_suggest_by_silhouette(self):
        """Test silhouette-based k selection"""
        optimizer = ClusterOptimizer(
            geocode_dataframe=self.geocode_dataframe,
            distance_matrix=self.distance_matrix,
            connectivity_matrix=self.connectivity_matrix,
        )

        metrics_list = optimizer.evaluate_k_range(k_min=2, k_max=5)
        result = optimizer.suggest_optimal_k(metrics_list, method="silhouette")

        self.assertEqual(result.selection_method, "silhouette")
        # Optimal k should be the one with highest silhouette
        max_silhouette_k = max(metrics_list, key=lambda m: m.silhouette).k
        self.assertEqual(result.optimal_k, max_silhouette_k)

    def test_suggest_by_elbow(self):
        """Test elbow method k selection"""
        optimizer = ClusterOptimizer(
            geocode_dataframe=self.geocode_dataframe,
            distance_matrix=self.distance_matrix,
            connectivity_matrix=self.connectivity_matrix,
        )

        metrics_list = optimizer.evaluate_k_range(k_min=2, k_max=5)
        result = optimizer.suggest_optimal_k(metrics_list, method="elbow")

        self.assertEqual(result.selection_method, "elbow")
        self.assertIn(result.optimal_k, [m.k for m in metrics_list])

    def test_suggest_by_compromise(self):
        """Test compromise method k selection"""
        optimizer = ClusterOptimizer(
            geocode_dataframe=self.geocode_dataframe,
            distance_matrix=self.distance_matrix,
            connectivity_matrix=self.connectivity_matrix,
        )

        metrics_list = optimizer.evaluate_k_range(k_min=2, k_max=5)
        result = optimizer.suggest_optimal_k(metrics_list, method="compromise")

        self.assertEqual(result.selection_method, "compromise")
        self.assertIn(result.optimal_k, [m.k for m in metrics_list])

    def test_suggest_by_multi_criteria(self):
        """Test multi-criteria k selection"""
        optimizer = ClusterOptimizer(
            geocode_dataframe=self.geocode_dataframe,
            distance_matrix=self.distance_matrix,
            connectivity_matrix=self.connectivity_matrix,
        )

        metrics_list = optimizer.evaluate_k_range(k_min=2, k_max=5)
        result = optimizer.suggest_optimal_k(metrics_list, method="multi_criteria")

        self.assertEqual(result.selection_method, "multi_criteria")
        self.assertIn(result.optimal_k, [m.k for m in metrics_list])
        self.assertIn("average rank", result.reason.lower())

    def test_metrics_to_dataframe(self):
        """Test conversion of metrics list to DataFrame"""
        metrics_list = [
            ClusterMetrics(
                k=2,
                silhouette=0.5,
                davies_bouldin=0.8,
                calinski_harabasz=100.0,
                inertia=50.0,
                mean_cluster_size=10,
                min_cluster_size=8,
                max_cluster_size=12,
            ),
            ClusterMetrics(
                k=3,
                silhouette=0.6,
                davies_bouldin=0.7,
                calinski_harabasz=110.0,
                inertia=40.0,
                mean_cluster_size=7,
                min_cluster_size=5,
                max_cluster_size=9,
            ),
        ]

        df = metrics_to_dataframe(metrics_list)

        self.assertIsInstance(df, pl.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertIn("k", df.columns)
        self.assertIn("silhouette", df.columns)
        self.assertIn("davies_bouldin", df.columns)
        self.assertEqual(df["k"].to_list(), [2, 3])
        self.assertEqual(df["silhouette"].to_list(), [0.5, 0.6])

    def test_create_metrics_report(self):
        """Test metrics report generation"""
        metrics = ClusterMetrics(
            k=5,
            silhouette=0.5,
            davies_bouldin=0.8,
            calinski_harabasz=100.0,
            inertia=50.0,
            mean_cluster_size=4,
            min_cluster_size=2,
            max_cluster_size=6,
        )

        result = OptimalKResult(
            optimal_k=5,
            all_metrics=[metrics],
            selection_method="test_method",
            reason="test reason",
        )

        report = create_metrics_report(result)

        self.assertIsInstance(report, str)
        self.assertIn("CLUSTER OPTIMIZATION REPORT", report)
        self.assertIn("Optimal k: 5", report)
        self.assertIn("test_method", report)
        self.assertIn("test reason", report)
        self.assertIn("Silhouette", report)
        self.assertIn("Davies-Bouldin", report)

    def test_ranking_functions(self):
        """Test ranking helper functions"""
        values = [1.0, 5.0, 3.0, 2.0]

        # Test ascending rank (higher is better)
        ranks_asc = ClusterOptimizer._rank_ascending(values)
        self.assertEqual(len(ranks_asc), 4)
        # Highest value (5.0) should have rank 1.0
        self.assertEqual(ranks_asc[1], 1.0)
        # Lowest value (1.0) should have rank 0.0
        self.assertEqual(ranks_asc[0], 0.0)

        # Test descending rank (lower is better)
        ranks_desc = ClusterOptimizer._rank_descending(values)
        self.assertEqual(len(ranks_desc), 4)
        # Lowest value (1.0) should have rank 1.0
        self.assertEqual(ranks_desc[0], 1.0)
        # Highest value (5.0) should have rank 0.0
        self.assertEqual(ranks_desc[1], 0.0)

    def test_inertia_calculation(self):
        """Test inertia calculation"""
        optimizer = ClusterOptimizer(
            geocode_dataframe=self.geocode_dataframe,
            distance_matrix=self.distance_matrix,
            connectivity_matrix=self.connectivity_matrix,
        )

        # Create simple labels
        labels = np.array([0, 0, 0, 1, 1, 1] + [2] * 14)

        inertia = optimizer._calculate_inertia(labels)

        self.assertIsInstance(inertia, float)
        self.assertGreater(inertia, 0)

    def test_metrics_values_reasonable(self):
        """Test that calculated metrics have reasonable values"""
        optimizer = ClusterOptimizer(
            geocode_dataframe=self.geocode_dataframe,
            distance_matrix=self.distance_matrix,
            connectivity_matrix=self.connectivity_matrix,
        )

        metrics_list = optimizer.evaluate_k_range(k_min=2, k_max=4)

        for metrics in metrics_list:
            # Silhouette should be in [-1, 1]
            self.assertGreaterEqual(metrics.silhouette, -1.0)
            self.assertLessEqual(metrics.silhouette, 1.0)

            # Davies-Bouldin should be non-negative
            self.assertGreaterEqual(metrics.davies_bouldin, 0.0)

            # Calinski-Harabasz should be positive
            self.assertGreater(metrics.calinski_harabasz, 0.0)

            # Inertia should be positive
            self.assertGreater(metrics.inertia, 0.0)

            # Cluster sizes should be positive
            self.assertGreater(metrics.min_cluster_size, 0)
            self.assertGreater(metrics.mean_cluster_size, 0)
            self.assertGreaterEqual(metrics.max_cluster_size, metrics.min_cluster_size)

    def test_all_methods_produce_valid_results(self):
        """Test that all suggestion methods produce valid results"""
        optimizer = ClusterOptimizer(
            geocode_dataframe=self.geocode_dataframe,
            distance_matrix=self.distance_matrix,
            connectivity_matrix=self.connectivity_matrix,
        )

        metrics_list = optimizer.evaluate_k_range(k_min=2, k_max=5)

        methods = ["silhouette", "elbow", "compromise", "multi_criteria"]

        for method in methods:
            with self.subTest(method=method):
                result = optimizer.suggest_optimal_k(metrics_list, method=method)

                self.assertEqual(result.selection_method, method)
                self.assertIn(result.optimal_k, [m.k for m in metrics_list])
                self.assertIsInstance(result.reason, str)
                self.assertGreater(len(result.reason), 0)


if __name__ == "__main__":
    unittest.main()
