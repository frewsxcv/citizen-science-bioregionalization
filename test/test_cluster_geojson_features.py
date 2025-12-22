import unittest

from src.dataframes.cluster_geojson_features import (
    ClusterGeojsonFeaturesSchema,
    get_feature_for_cluster,
    to_feature_collection,
)
from src.geojson import (
    build_geojson_feature_collection,
)
from test.fixtures.cluster_boundary import mock_cluster_boundary_df
from test.fixtures.cluster_color import mock_cluster_color_df


class TestClusterGeojsonFeaturesDataFrame(unittest.TestCase):
    def setUp(self):
        self.cluster_boundary_df = mock_cluster_boundary_df()
        self.cluster_colors_df = mock_cluster_color_df()

    def test_build_cluster_geojson_features_df(self):
        # Build the dataframe
        cluster_geojson_features_df = ClusterGeojsonFeaturesSchema.build_df(
            self.cluster_boundary_df, self.cluster_colors_df
        )

        # Verify we have the expected number of rows
        self.assertEqual(2, cluster_geojson_features_df.height)

        # Verify the clusters
        clusters = cluster_geojson_features_df["cluster"].to_list()
        self.assertEqual([1, 2], clusters)

    def test_get_feature_for_cluster(self):
        # Build the dataframe
        cluster_geojson_features_df = ClusterGeojsonFeaturesSchema.build_df(
            self.cluster_boundary_df, self.cluster_colors_df
        )

        # Get a feature for a specific cluster
        feature = get_feature_for_cluster(cluster_geojson_features_df, 1)

        # Verify the feature
        assert feature is not None
        self.assertEqual("Feature", feature["type"])
        self.assertEqual(1, feature["properties"]["cluster"])
        self.assertEqual("#ff0000", feature["properties"]["fillColor"])
        self.assertEqual("#800000", feature["properties"]["color"])

        # Test getting a non-existent cluster
        feature = get_feature_for_cluster(cluster_geojson_features_df, 999)
        self.assertIsNone(feature)

    def test_to_feature_collection(self):
        # Build the dataframe
        cluster_geojson_features_df = ClusterGeojsonFeaturesSchema.build_df(
            self.cluster_boundary_df, self.cluster_colors_df
        )

        # Convert to feature collection
        feature_collection = to_feature_collection(cluster_geojson_features_df)

        # Verify the feature collection
        self.assertEqual("FeatureCollection", feature_collection["type"])
        self.assertEqual(2, len(feature_collection["features"]))

        # The geojson feature_collection should match what we'd get from build_geojson_feature_collection
        original_feature_collection = build_geojson_feature_collection(
            self.cluster_boundary_df, self.cluster_colors_df
        )

        # Compare the two feature collections (they should be equivalent)
        self.assertEqual(original_feature_collection, feature_collection)


if __name__ == "__main__":
    unittest.main()
