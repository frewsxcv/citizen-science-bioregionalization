import unittest

import numpy as np
import polars as pl

from src.dataframes.geocode_silhouette_score import (
    GeocodeSilhouetteScoreSchema,
    build_geocode_silhouette_score_df,
)
from src.matrices.geocode_distance import GeocodeDistanceMatrix
from test.fixtures.geocode_cluster import mock_geocode_cluster_multi_k_df


class TestGeocodeSilhouetteScore(unittest.TestCase):
    def test_build_df_with_num_clusters(self):
        """
        Test that build_df correctly includes num_clusters field.
        """
        # Create a condensed distance matrix for 3 geocodes
        # Condensed format contains pairwise distances: [d(0,1), d(0,2), d(1,2)]
        condensed_distances = np.array([0.5, 0.9, 0.8])
        distance_matrix = GeocodeDistanceMatrix(condensed_distances)

        num_clusters = 2
        geocode_cluster_df = mock_geocode_cluster_multi_k_df(num_clusters=num_clusters)

        result = build_geocode_silhouette_score_df(
            distance_matrix,
            geocode_cluster_df,
        )

        # Verify schema
        self.assertIn("geocode", result.columns)
        self.assertIn("silhouette_score", result.columns)
        self.assertIn("num_clusters", result.columns)

        # Verify num_clusters is populated for all rows
        self.assertTrue((result["num_clusters"] == num_clusters).all())

        # Verify we have overall score (geocode=null) plus per-geocode scores
        overall_score = result.filter(pl.col("geocode").is_null())
        self.assertEqual(len(overall_score), 1)
        self.assertEqual(overall_score["num_clusters"][0], num_clusters)

        per_geocode_scores = result.filter(pl.col("geocode").is_not_null())
        self.assertEqual(len(per_geocode_scores), 3)  # 3 geocodes
        self.assertTrue((per_geocode_scores["num_clusters"] == num_clusters).all())

    def test_build_df_different_num_clusters(self):
        """
        Test that different num_clusters values are correctly stored.
        """
        # Create a condensed distance matrix for 3 geocodes
        condensed_distances = np.array([0.5, 0.9, 0.8])
        distance_matrix = GeocodeDistanceMatrix(condensed_distances)

        # Test with different num_clusters values
        for num_k in [2, 5, 10]:
            geocode_cluster_df = mock_geocode_cluster_multi_k_df(num_clusters=num_k)
            result = build_geocode_silhouette_score_df(
                distance_matrix,
                geocode_cluster_df,
            )

            self.assertTrue((result["num_clusters"] == num_k).all())

    def test_filter_by_num_clusters(self):
        """
        Test that we can filter results by num_clusters.
        """
        # Create a condensed distance matrix for 3 geocodes
        condensed_distances = np.array([0.5, 0.9, 0.8])
        distance_matrix = GeocodeDistanceMatrix(condensed_distances)

        # Build two dataframes with different num_clusters
        geocode_cluster_df_2 = mock_geocode_cluster_multi_k_df(num_clusters=2)
        result_2 = build_geocode_silhouette_score_df(
            distance_matrix, geocode_cluster_df_2
        )
        geocode_cluster_df_5 = mock_geocode_cluster_multi_k_df(num_clusters=5)
        result_5 = build_geocode_silhouette_score_df(
            distance_matrix, geocode_cluster_df_5
        )

        # Combine them
        combined = pl.concat([result_2, result_5])

        # Filter by num_clusters
        filtered_2 = combined.filter(pl.col("num_clusters") == 2)
        filtered_5 = combined.filter(pl.col("num_clusters") == 5)

        self.assertEqual(len(filtered_2), len(result_2))
        self.assertEqual(len(filtered_5), len(result_5))
        self.assertTrue((filtered_2["num_clusters"] == 2).all())
        self.assertTrue((filtered_5["num_clusters"] == 5).all())

    def test_schema_validation(self):
        """
        Test that schema validation works correctly with num_clusters field.
        """
        # Create a condensed distance matrix for 3 geocodes
        condensed_distances = np.array([0.5, 0.9, 0.8])
        distance_matrix = GeocodeDistanceMatrix(condensed_distances)
        geocode_cluster_df = mock_geocode_cluster_multi_k_df(num_clusters=3)

        result = build_geocode_silhouette_score_df(
            distance_matrix,
            geocode_cluster_df,
        )

        # Verify data types
        self.assertEqual(result.schema["geocode"], pl.UInt64)
        self.assertEqual(result.schema["silhouette_score"], pl.Float64)
        self.assertEqual(result.schema["num_clusters"], pl.UInt32)


if __name__ == "__main__":
    unittest.main()
