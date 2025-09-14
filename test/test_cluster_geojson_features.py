import unittest
import geojson  # type: ignore
import dataframely as dy

from src.dataframes.cluster_geojson_features import ClusterGeojsonFeaturesDataFrame
from src.geojson import (
    build_geojson_feature_collection,
)
from test.fixtures.cluster_boundary import mock_cluster_boundary_dataframe
from test.fixtures.cluster_color import mock_cluster_color_dataframe


class TestClusterGeojsonFeaturesDataFrame(unittest.TestCase):
    def setUp(self):
        self.cluster_boundary_dataframe = mock_cluster_boundary_dataframe()
        self.cluster_colors_dataframe = mock_cluster_color_dataframe()

    def test_build_cluster_geojson_features_dataframe(self):
        # Build the dataframe
        cluster_geojson_features_dataframe = ClusterGeojsonFeaturesDataFrame.build(
            self.cluster_boundary_dataframe, self.cluster_colors_dataframe
        )

        # Verify we have the expected number of rows
        self.assertEqual(2, cluster_geojson_features_dataframe.df.height)

        # Verify the schema is correct
        self.assertEqual(
            ClusterGeojsonFeaturesDataFrame.SCHEMA,
            cluster_geojson_features_dataframe.df.schema,
        )

        # Verify the clusters
        clusters = cluster_geojson_features_dataframe.df["cluster"].to_list()
        self.assertEqual([1, 2], clusters)

    def test_get_feature_for_cluster(self):
        # Build the dataframe
        cluster_geojson_features_dataframe = ClusterGeojsonFeaturesDataFrame.build(
            self.cluster_boundary_dataframe, self.cluster_colors_dataframe
        )

        # Get a feature for a specific cluster
        feature = cluster_geojson_features_dataframe.get_feature_for_cluster(1)

        # Verify the feature
        assert feature is not None
        self.assertEqual("Feature", feature["type"])
        self.assertEqual(1, feature["properties"]["cluster"])
        self.assertEqual("#ff0000", feature["properties"]["fillColor"])
        self.assertEqual("#800000", feature["properties"]["color"])

        # Test getting a non-existent cluster
        feature = cluster_geojson_features_dataframe.get_feature_for_cluster(999)
        self.assertIsNone(feature)

    def test_to_feature_collection(self):
        # Build the dataframe
        cluster_geojson_features_dataframe = ClusterGeojsonFeaturesDataFrame.build(
            self.cluster_boundary_dataframe, self.cluster_colors_dataframe
        )

        # Convert to feature collection
        feature_collection = cluster_geojson_features_dataframe.to_feature_collection()

        # Verify the feature collection
        self.assertEqual("FeatureCollection", feature_collection["type"])
        self.assertEqual(2, len(feature_collection["features"]))

        # The geojson feature_collection should match what we'd get from build_geojson_feature_collection
        original_feature_collection = build_geojson_feature_collection(
            self.cluster_boundary_dataframe, self.cluster_colors_dataframe
        )

        # Compare the two feature collections (they should be equivalent)
        self.assertEqual(original_feature_collection, feature_collection)


if __name__ == "__main__":
    unittest.main()
