import unittest
import polars as pl
import shapely
import geojson  # type: ignore
import dataframely as dy

from src.dataframes.cluster_boundary import ClusterBoundarySchema
from src.dataframes.cluster_color import ClusterColorDataFrame
from src.dataframes.cluster_geojson_features import ClusterGeojsonFeaturesDataFrame
from src.geojson import (
    build_geojson_feature_collection,
)


class TestClusterGeojsonFeaturesDataFrame(unittest.TestCase):
    def setUp(self):
        # Create mock polygon geometries
        self.polygon1 = shapely.Polygon(
            [
                (-67.762704, 49.598604),
                (-68.147336, 49.56361),
                (-68.295514, 49.333781),
                (-68.061716, 49.140104),
                (-67.680942, 49.174866),
                (-67.530135, 49.403534),
                (-67.762704, 49.598604),
            ]
        )
        self.polygon2 = shapely.Polygon(
            [
                (-69.201662, 49.028234),
                (-69.342511, 48.797701),
                (-69.719479, 48.757432),
                (-69.959318, 48.947367),
                (-69.821222, 49.178989),
                (-70.063974, 49.369372),
                (-69.925042, 49.60178),
                (-69.540512, 49.642718),
                (-69.298758, 49.450906),
                (-69.44051, 49.219589),
                (-69.201662, 49.028234),
            ]
        )

        # Create a mock ClusterBoundaryDataFrame
        cluster_boundary_df = pl.DataFrame(
            [
                {"cluster": 1, "geometry": shapely.to_wkb(self.polygon1)},
                {"cluster": 2, "geometry": shapely.to_wkb(self.polygon2)},
            ]
        ).with_columns(pl.col("cluster").cast(pl.UInt32()))

        self.cluster_boundary_dataframe = ClusterBoundarySchema.validate(
            cluster_boundary_df
        )

        # Create a mock ClusterColorDataFrame
        self.cluster_colors_dataframe = ClusterColorDataFrame(
            df=pl.DataFrame(
                [
                    {"cluster": 1, "color": "#ff0000", "darkened_color": "#800000"},
                    {"cluster": 2, "color": "#0000ff", "darkened_color": "#000080"},
                ],
                schema=ClusterColorDataFrame.SCHEMA,
            )
        )

        # Expected feature collection data
        self.expected_feature_collection = {
            "features": [
                {
                    "geometry": {
                        "coordinates": [
                            [
                                [-67.762704, 49.598604],
                                [-68.147336, 49.56361],
                                [-68.295514, 49.333781],
                                [-68.061716, 49.140104],
                                [-67.680942, 49.174866],
                                [-67.530135, 49.403534],
                                [-67.762704, 49.598604],
                            ]
                        ],
                        "type": "Polygon",
                    },
                    "properties": {
                        "cluster": 1,
                        "color": "#800000",
                        "fillColor": "#ff0000",
                        "fillOpacity": 0.7,
                        "weight": 1,
                    },
                    "type": "Feature",
                },
                {
                    "geometry": {
                        "coordinates": [
                            [
                                [-69.201662, 49.028234],
                                [-69.342511, 48.797701],
                                [-69.719479, 48.757432],
                                [-69.959318, 48.947367],
                                [-69.821222, 49.178989],
                                [-70.063974, 49.369372],
                                [-69.925042, 49.60178],
                                [-69.540512, 49.642718],
                                [-69.298758, 49.450906],
                                [-69.44051, 49.219589],
                                [-69.201662, 49.028234],
                            ]
                        ],
                        "type": "Polygon",
                    },
                    "properties": {
                        "cluster": 2,
                        "color": "#000080",
                        "fillColor": "#0000ff",
                        "fillOpacity": 0.7,
                        "weight": 1,
                    },
                    "type": "Feature",
                },
            ],
            "type": "FeatureCollection",
        }

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
        self.assertIsNotNone(feature)
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
