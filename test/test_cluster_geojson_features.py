import unittest

from src.geojson import build_geojson_feature_collection
from test.fixtures.cluster_boundary import mock_cluster_boundary_df
from test.fixtures.cluster_color import mock_cluster_color_df


class TestGeojsonFeatures(unittest.TestCase):
    def setUp(self):
        self.cluster_boundary_df = mock_cluster_boundary_df()
        self.cluster_colors_df = mock_cluster_color_df()

    def test_build_geojson_feature_collection(self):
        # Build the feature collection directly
        feature_collection = build_geojson_feature_collection(
            self.cluster_boundary_df, self.cluster_colors_df
        )

        # Verify the feature collection structure
        self.assertEqual("FeatureCollection", feature_collection["type"])
        self.assertEqual(2, len(feature_collection["features"]))

        # Get features sorted by cluster for consistent testing
        features = sorted(
            feature_collection["features"],
            key=lambda f: f["properties"]["cluster"],
        )

        # Verify first feature (cluster 1)
        feature1 = features[0]
        self.assertEqual("Feature", feature1["type"])
        self.assertEqual(1, feature1["properties"]["cluster"])
        self.assertEqual("#ff0000", feature1["properties"]["fillColor"])
        self.assertEqual("#800000", feature1["properties"]["color"])
        self.assertEqual(0.7, feature1["properties"]["fillOpacity"])
        self.assertEqual(1, feature1["properties"]["weight"])

        # Verify second feature (cluster 2)
        feature2 = features[1]
        self.assertEqual("Feature", feature2["type"])
        self.assertEqual(2, feature2["properties"]["cluster"])
        self.assertEqual("#0000ff", feature2["properties"]["fillColor"])
        self.assertEqual("#000080", feature2["properties"]["color"])

    def test_build_geojson_feature_collection_has_geometries(self):
        # Build the feature collection
        feature_collection = build_geojson_feature_collection(
            self.cluster_boundary_df, self.cluster_colors_df
        )

        # Verify each feature has a geometry
        for feature in feature_collection["features"]:
            self.assertIn("geometry", feature)
            self.assertIsNotNone(feature["geometry"])
            self.assertIn("type", feature["geometry"])
            self.assertIn("coordinates", feature["geometry"])


if __name__ == "__main__":
    unittest.main()
