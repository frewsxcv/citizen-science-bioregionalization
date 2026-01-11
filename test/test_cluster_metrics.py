"""
Unit tests for multi-metric cluster validation.

Tests the GeocodeClusterMetricsSchema, builder functions, and selection logic.
"""

import unittest

import dataframely as dy
import numpy as np
import polars as pl

from src.dataframes.geocode_cluster import GeocodeClusterMultiKSchema
from src.dataframes.geocode_cluster_metrics import (
    GeocodeClusterMetricsSchema,
    build_geocode_cluster_metrics_df,
    get_metric_interpretations,
    get_metrics_summary,
    select_optimal_k_multi_metric,
)
from src.matrices.geocode_distance import GeocodeDistanceMatrix


def mock_geocode_cluster_multi_k_df(
    num_geocodes: int = 10,
    k_values: list[int] | None = None,
) -> dy.DataFrame[GeocodeClusterMultiKSchema]:
    """Create a mock multi-k clustering DataFrame for testing."""
    if k_values is None:
        k_values = [2, 3, 4, 5]

    all_rows = []
    for k in k_values:
        # Generate cluster assignments (cycling through 0 to k-1)
        clusters = [i % k for i in range(num_geocodes)]
        for i, cluster in enumerate(clusters):
            all_rows.append(
                {
                    "geocode": i + 1,  # 1-indexed geocodes
                    "num_clusters": k,
                    "cluster": cluster,
                }
            )

    df = pl.DataFrame(all_rows).with_columns(
        pl.col("geocode").cast(pl.UInt64),
        pl.col("num_clusters").cast(pl.UInt32),
        pl.col("cluster").cast(pl.UInt32),
    )

    return GeocodeClusterMultiKSchema.validate(df)


def mock_distance_matrix(num_geocodes: int = 10) -> GeocodeDistanceMatrix:
    """Create a mock distance matrix for testing."""
    # Create a random but symmetric distance matrix
    np.random.seed(42)  # For reproducibility
    random_matrix = np.random.rand(num_geocodes, num_geocodes)
    symmetric_matrix = (random_matrix + random_matrix.T) / 2
    np.fill_diagonal(symmetric_matrix, 0)

    # Convert to condensed form
    from scipy.spatial.distance import squareform

    condensed = squareform(symmetric_matrix)

    # Create reduced features (mock UMAP output)
    reduced_features = np.random.rand(num_geocodes, 5)

    return GeocodeDistanceMatrix(condensed, reduced_features)


class TestGeocodeClusterMetricsSchema(unittest.TestCase):
    """Tests for GeocodeClusterMetricsSchema validation."""

    def test_schema_validates_valid_data(self):
        """Test that valid data passes schema validation."""
        df = pl.DataFrame(
            {
                "num_clusters": [2, 3, 4],
                "silhouette_score": [0.5, 0.6, 0.4],
                "calinski_harabasz_score": [100.0, 150.0, 120.0],
                "davies_bouldin_score": [0.8, 0.6, 0.9],
                "inertia": [500.0, 300.0, 200.0],
                "silhouette_normalized": [0.75, 0.8, 0.7],
                "calinski_harabasz_normalized": [0.0, 1.0, 0.4],
                "davies_bouldin_normalized": [0.33, 1.0, 0.0],
                "inertia_normalized": [0.0, 0.67, 1.0],
                "combined_score": [0.6, 0.8, 0.5],
            }
        ).with_columns(pl.col("num_clusters").cast(pl.UInt32))

        validated = GeocodeClusterMetricsSchema.validate(df)
        self.assertEqual(len(validated), 3)

    def test_schema_requires_num_clusters(self):
        """Test that num_clusters column is required."""
        df = pl.DataFrame(
            {
                "silhouette_score": [0.5],
                "calinski_harabasz_score": [100.0],
                "davies_bouldin_score": [0.8],
                "inertia": [500.0],
                "silhouette_normalized": [0.75],
                "calinski_harabasz_normalized": [0.5],
                "davies_bouldin_normalized": [0.5],
                "inertia_normalized": [0.5],
                "combined_score": [0.6],
            }
        )

        with self.assertRaises(Exception):
            GeocodeClusterMetricsSchema.validate(df)


class TestBuildGeocodeClusterMetricsDf(unittest.TestCase):
    """Tests for build_geocode_cluster_metrics_df function."""

    def test_builds_metrics_for_all_k_values(self):
        """Test that metrics are computed for all k values in input."""
        num_geocodes = 20
        k_values = [2, 3, 4, 5, 6]

        cluster_df = mock_geocode_cluster_multi_k_df(
            num_geocodes=num_geocodes, k_values=k_values
        )
        distance_matrix = mock_distance_matrix(num_geocodes=num_geocodes)

        metrics_df = build_geocode_cluster_metrics_df(distance_matrix, cluster_df)

        # Should have one row per k value
        self.assertEqual(len(metrics_df), len(k_values))
        self.assertEqual(
            sorted(metrics_df["num_clusters"].to_list()), sorted(k_values)
        )

    def test_silhouette_scores_in_valid_range(self):
        """Test that silhouette scores are in [-1, 1] range."""
        num_geocodes = 20
        k_values = [2, 3, 4]

        cluster_df = mock_geocode_cluster_multi_k_df(
            num_geocodes=num_geocodes, k_values=k_values
        )
        distance_matrix = mock_distance_matrix(num_geocodes=num_geocodes)

        metrics_df = build_geocode_cluster_metrics_df(distance_matrix, cluster_df)

        sil_scores = metrics_df["silhouette_score"].to_list()
        for score in sil_scores:
            self.assertGreaterEqual(score, -1.0)
            self.assertLessEqual(score, 1.0)

    def test_calinski_harabasz_positive(self):
        """Test that Calinski-Harabasz scores are positive."""
        num_geocodes = 20
        k_values = [2, 3, 4]

        cluster_df = mock_geocode_cluster_multi_k_df(
            num_geocodes=num_geocodes, k_values=k_values
        )
        distance_matrix = mock_distance_matrix(num_geocodes=num_geocodes)

        metrics_df = build_geocode_cluster_metrics_df(distance_matrix, cluster_df)

        ch_scores = metrics_df["calinski_harabasz_score"].to_list()
        for score in ch_scores:
            self.assertGreater(score, 0)

    def test_davies_bouldin_positive(self):
        """Test that Davies-Bouldin scores are positive."""
        num_geocodes = 20
        k_values = [2, 3, 4]

        cluster_df = mock_geocode_cluster_multi_k_df(
            num_geocodes=num_geocodes, k_values=k_values
        )
        distance_matrix = mock_distance_matrix(num_geocodes=num_geocodes)

        metrics_df = build_geocode_cluster_metrics_df(distance_matrix, cluster_df)

        db_scores = metrics_df["davies_bouldin_score"].to_list()
        for score in db_scores:
            self.assertGreater(score, 0)

    def test_normalized_scores_in_zero_one_range(self):
        """Test that normalized scores are in [0, 1] range."""
        num_geocodes = 20
        k_values = [2, 3, 4, 5]

        cluster_df = mock_geocode_cluster_multi_k_df(
            num_geocodes=num_geocodes, k_values=k_values
        )
        distance_matrix = mock_distance_matrix(num_geocodes=num_geocodes)

        metrics_df = build_geocode_cluster_metrics_df(distance_matrix, cluster_df)

        for col in [
            "silhouette_normalized",
            "calinski_harabasz_normalized",
            "davies_bouldin_normalized",
            "combined_score",
        ]:
            scores = metrics_df[col].to_list()
            for score in scores:
                self.assertGreaterEqual(
                    score, 0.0, f"{col} has score {score} below 0"
                )
                self.assertLessEqual(score, 1.0, f"{col} has score {score} above 1")

    def test_custom_weights(self):
        """Test that custom weights are applied correctly."""
        num_geocodes = 20
        k_values = [2, 3]

        cluster_df = mock_geocode_cluster_multi_k_df(
            num_geocodes=num_geocodes, k_values=k_values
        )
        distance_matrix = mock_distance_matrix(num_geocodes=num_geocodes)

        # Use weights that heavily favor silhouette
        weights = {"silhouette": 0.8, "calinski_harabasz": 0.1, "davies_bouldin": 0.1}

        metrics_df = build_geocode_cluster_metrics_df(
            distance_matrix, cluster_df, weights=weights
        )

        # Combined score should be closer to silhouette_normalized
        for row in metrics_df.iter_rows(named=True):
            combined = row["combined_score"]
            sil_norm = row["silhouette_normalized"]
            # With 0.8 weight, combined should be within 0.2 of silhouette_normalized
            self.assertAlmostEqual(combined, sil_norm, delta=0.25)


class TestSelectOptimalKMultiMetric(unittest.TestCase):
    """Tests for select_optimal_k_multi_metric function."""

    def create_metrics_df(
        self, scores: list[dict[str, float]]
    ) -> dy.DataFrame[GeocodeClusterMetricsSchema]:
        """Helper to create a metrics DataFrame from score dicts."""
        df = pl.DataFrame(scores).with_columns(
            pl.col("num_clusters").cast(pl.UInt32)
        )
        return GeocodeClusterMetricsSchema.validate(df)

    def test_selects_highest_combined_score(self):
        """Test that 'combined' method selects k with highest combined score."""
        metrics_df = self.create_metrics_df(
            [
                {
                    "num_clusters": 2,
                    "silhouette_score": 0.3,
                    "calinski_harabasz_score": 100.0,
                    "davies_bouldin_score": 0.8,
                    "inertia": 500.0,
                    "silhouette_normalized": 0.65,
                    "calinski_harabasz_normalized": 0.0,
                    "davies_bouldin_normalized": 1.0,
                    "inertia_normalized": 0.0,
                    "combined_score": 0.5,
                },
                {
                    "num_clusters": 3,
                    "silhouette_score": 0.5,
                    "calinski_harabasz_score": 150.0,
                    "davies_bouldin_score": 0.6,
                    "inertia": 300.0,
                    "silhouette_normalized": 0.75,
                    "calinski_harabasz_normalized": 1.0,
                    "davies_bouldin_normalized": 0.5,
                    "inertia_normalized": 0.67,
                    "combined_score": 0.75,  # Highest
                },
                {
                    "num_clusters": 4,
                    "silhouette_score": 0.4,
                    "calinski_harabasz_score": 120.0,
                    "davies_bouldin_score": 0.7,
                    "inertia": 200.0,
                    "silhouette_normalized": 0.7,
                    "calinski_harabasz_normalized": 0.4,
                    "davies_bouldin_normalized": 0.75,
                    "inertia_normalized": 1.0,
                    "combined_score": 0.6,
                },
            ]
        )

        optimal_k = select_optimal_k_multi_metric(
            metrics_df, min_silhouette_threshold=None, selection_method="combined"
        )

        self.assertEqual(optimal_k, 3)

    def test_respects_silhouette_threshold(self):
        """Test that silhouette threshold filters out low-quality k values."""
        metrics_df = self.create_metrics_df(
            [
                {
                    "num_clusters": 2,
                    "silhouette_score": 0.2,  # Below threshold
                    "calinski_harabasz_score": 100.0,
                    "davies_bouldin_score": 0.8,
                    "inertia": 500.0,
                    "silhouette_normalized": 0.6,
                    "calinski_harabasz_normalized": 0.0,
                    "davies_bouldin_normalized": 1.0,
                    "inertia_normalized": 0.0,
                    "combined_score": 0.9,  # Highest combined but low silhouette
                },
                {
                    "num_clusters": 3,
                    "silhouette_score": 0.3,  # Above threshold
                    "calinski_harabasz_score": 80.0,
                    "davies_bouldin_score": 0.9,
                    "inertia": 300.0,
                    "silhouette_normalized": 0.65,
                    "calinski_harabasz_normalized": 0.5,
                    "davies_bouldin_normalized": 0.5,
                    "inertia_normalized": 1.0,
                    "combined_score": 0.6,
                },
            ]
        )

        optimal_k = select_optimal_k_multi_metric(
            metrics_df, min_silhouette_threshold=0.25, selection_method="combined"
        )

        # Should select k=3 because k=2 is filtered out
        self.assertEqual(optimal_k, 3)

    def test_returns_none_when_all_below_threshold(self):
        """Test that None is returned when no k meets threshold."""
        metrics_df = self.create_metrics_df(
            [
                {
                    "num_clusters": 2,
                    "silhouette_score": 0.1,  # Below threshold
                    "calinski_harabasz_score": 100.0,
                    "davies_bouldin_score": 0.8,
                    "inertia": 500.0,
                    "silhouette_normalized": 0.55,
                    "calinski_harabasz_normalized": 0.0,
                    "davies_bouldin_normalized": 0.0,
                    "inertia_normalized": 0.0,
                    "combined_score": 0.3,
                },
                {
                    "num_clusters": 3,
                    "silhouette_score": 0.15,  # Below threshold
                    "calinski_harabasz_score": 150.0,
                    "davies_bouldin_score": 0.6,
                    "inertia": 300.0,
                    "silhouette_normalized": 0.575,
                    "calinski_harabasz_normalized": 1.0,
                    "davies_bouldin_normalized": 1.0,
                    "inertia_normalized": 1.0,
                    "combined_score": 0.8,
                },
            ]
        )

        optimal_k = select_optimal_k_multi_metric(
            metrics_df, min_silhouette_threshold=0.25, selection_method="combined"
        )

        self.assertIsNone(optimal_k)

    def test_silhouette_only_method(self):
        """Test that 'silhouette' method uses only silhouette score."""
        metrics_df = self.create_metrics_df(
            [
                {
                    "num_clusters": 2,
                    "silhouette_score": 0.6,  # Highest silhouette
                    "calinski_harabasz_score": 50.0,  # Lowest CH
                    "davies_bouldin_score": 1.5,  # Worst DB
                    "inertia": 500.0,
                    "silhouette_normalized": 0.8,
                    "calinski_harabasz_normalized": 0.0,
                    "davies_bouldin_normalized": 0.0,
                    "inertia_normalized": 0.0,
                    "combined_score": 0.3,  # Lowest combined
                },
                {
                    "num_clusters": 3,
                    "silhouette_score": 0.4,
                    "calinski_harabasz_score": 200.0,
                    "davies_bouldin_score": 0.5,
                    "inertia": 300.0,
                    "silhouette_normalized": 0.7,
                    "calinski_harabasz_normalized": 1.0,
                    "davies_bouldin_normalized": 1.0,
                    "inertia_normalized": 1.0,
                    "combined_score": 0.9,  # Highest combined
                },
            ]
        )

        optimal_k = select_optimal_k_multi_metric(
            metrics_df, min_silhouette_threshold=None, selection_method="silhouette"
        )

        # Should select k=2 based on silhouette alone
        self.assertEqual(optimal_k, 2)

class TestGetMetricsSummary(unittest.TestCase):
    """Tests for get_metrics_summary function."""

    def test_returns_sorted_by_combined_score(self):
        """Test that summary is sorted by combined score descending."""
        df = pl.DataFrame(
            {
                "num_clusters": [2, 5, 10],
                "silhouette_score": [0.5, 0.45, 0.4],
                "calinski_harabasz_score": [200.0, 150.0, 100.0],
                "davies_bouldin_score": [0.5, 0.6, 0.7],
                "inertia": [800.0, 400.0, 200.0],
                "silhouette_normalized": [0.75, 0.725, 0.7],
                "calinski_harabasz_normalized": [1.0, 0.5, 0.0],
                "davies_bouldin_normalized": [1.0, 0.5, 0.0],
                "inertia_normalized": [0.0, 0.5, 1.0],
                "combined_score": [0.9, 0.6, 0.3],
            }
        ).with_columns(pl.col("num_clusters").cast(pl.UInt32))

        metrics_df = GeocodeClusterMetricsSchema.validate(df)
        summary = get_metrics_summary(metrics_df)

        # Should be sorted by combined_score descending
        self.assertEqual(summary["num_clusters"].to_list(), [2, 5, 10])
        self.assertEqual(summary["rank"].to_list(), [1, 2, 3])

    def test_includes_expected_columns(self):
        """Test that summary includes rank and key metrics."""
        df = pl.DataFrame(
            {
                "num_clusters": [2, 3],
                "silhouette_score": [0.3, 0.5],
                "calinski_harabasz_score": [100.0, 150.0],
                "davies_bouldin_score": [0.8, 0.6],
                "inertia": [500.0, 300.0],
                "silhouette_normalized": [0.65, 0.75],
                "calinski_harabasz_normalized": [0.0, 1.0],
                "davies_bouldin_normalized": [0.0, 1.0],
                "inertia_normalized": [0.0, 1.0],
                "combined_score": [0.3, 0.85],
            }
        ).with_columns(pl.col("num_clusters").cast(pl.UInt32))

        metrics_df = GeocodeClusterMetricsSchema.validate(df)
        summary = get_metrics_summary(metrics_df)

        expected_columns = {
            "rank",
            "num_clusters",
            "silhouette_score",
            "calinski_harabasz_score",
            "davies_bouldin_score",
            "inertia",
            "combined_score",
        }
        self.assertEqual(set(summary.columns), expected_columns)


class TestGetMetricInterpretations(unittest.TestCase):
    """Tests for get_metric_interpretations function."""

    def test_returns_interpretations_for_all_metrics(self):
        """Test that interpretations are provided for all metrics."""
        interpretations = get_metric_interpretations()

        expected_keys = {
            "silhouette_score",
            "calinski_harabasz_score",
            "davies_bouldin_score",
            "inertia",
            "combined_score",
        }
        self.assertEqual(set(interpretations.keys()), expected_keys)

    def test_interpretations_are_non_empty_strings(self):
        """Test that all interpretations are non-empty strings."""
        interpretations = get_metric_interpretations()

        for key, value in interpretations.items():
            self.assertIsInstance(value, str, f"{key} interpretation is not a string")
            self.assertGreater(
                len(value), 0, f"{key} interpretation is empty"
            )


if __name__ == "__main__":
    unittest.main()
