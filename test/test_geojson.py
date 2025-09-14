import unittest
import polars as pl
import shapely

from src.geojson import (
    build_geojson_feature_collection,
    is_cluster_mostly_ocean,
    find_ocean_clusters,
)
from src.dataframes.cluster_boundary import ClusterBoundarySchema
from src.dataframes.cluster_color import ClusterColorSchema


class TestGeojson(unittest.TestCase):
    def test_build_geojson_feature_collection(self):
        # Create mock polygon geometries
        polygon1 = shapely.Polygon(
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
        polygon2 = shapely.Polygon(
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
                {"cluster": 1, "geometry": shapely.to_wkb(polygon1)},
                {"cluster": 2, "geometry": shapely.to_wkb(polygon2)},
            ]
        ).with_columns(pl.col("cluster").cast(pl.UInt32()))

        cluster_boundary_dataframe = ClusterBoundarySchema.validate(cluster_boundary_df)

        actual = build_geojson_feature_collection(
            cluster_boundary_dataframe=cluster_boundary_dataframe,
            cluster_colors_dataframe=ClusterColorSchema.validate(
                pl.DataFrame(
                    [
                        {"cluster": 1, "color": "#ff0000", "darkened_color": "#800000"},
                        {"cluster": 2, "color": "#0000ff", "darkened_color": "#000080"},
                    ]
                ).with_columns(pl.col("cluster").cast(pl.UInt32()))
            ),
        )
        expected = {
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
        self.assertEqual(expected, actual)

    def test_ocean_clusters(self):
        """
        Test is_cluster_mostly_ocean and find_ocean_clusters functions using the actual ocean data.
        """
        # Create test cluster geometries
        # Test clusters located in different areas to test ocean overlap
        # Northeast US coastal area
        polygon1 = shapely.Polygon(
            [(-70.0, 43.0), (-70.0, 42.0), (-69.0, 42.0), (-69.0, 43.0), (-70.0, 43.0)]
        )

        # Inland US area (not ocean)
        polygon2 = shapely.Polygon(
            [(-80.0, 40.0), (-80.0, 39.0), (-79.0, 39.0), (-79.0, 40.0), (-80.0, 40.0)]
        )

        # Atlantic Ocean area
        polygon3 = shapely.Polygon(
            [(-65.0, 40.0), (-65.0, 39.0), (-64.0, 39.0), (-64.0, 40.0), (-65.0, 40.0)]
        )

        # Create a ClusterBoundaryDataFrame
        cluster_boundary_df = pl.DataFrame(
            [
                {"cluster": 1, "geometry": shapely.to_wkb(polygon1)},
                {"cluster": 2, "geometry": shapely.to_wkb(polygon2)},
                {"cluster": 3, "geometry": shapely.to_wkb(polygon3)},
            ]
        ).with_columns(pl.col("cluster").cast(pl.UInt32()))

        cluster_boundary_dataframe = ClusterBoundarySchema.validate(cluster_boundary_df)

        # Test is_cluster_mostly_ocean function with the real ocean file
        # Note: The actual results depend on the real ocean.geojson content
        # We expect cluster 3 to be in the ocean (Atlantic)
        self.assertTrue(
            is_cluster_mostly_ocean(cluster_boundary_dataframe, 3, threshold=0.5)
        )

        # Find ocean clusters with different thresholds
        # The actual results will depend on the real ocean data
        # We'll use assertIn/assertNotIn to flexibly test the expected behavior

        # Test with a moderate threshold
        ocean_clusters = find_ocean_clusters(cluster_boundary_dataframe, threshold=0.5)
        self.assertIn(
            3, ocean_clusters, "Cluster 3 should be identified as an ocean cluster"
        )

        # With a very high threshold (0.99), we may or may not get any clusters
        # depending on the exact ocean data
        ocean_clusters_high = find_ocean_clusters(
            cluster_boundary_dataframe, threshold=0.99
        )

        # With a very low threshold, we should at least get cluster 3
        ocean_clusters_low = find_ocean_clusters(
            cluster_boundary_dataframe, threshold=0.01
        )
        self.assertIn(
            3,
            ocean_clusters_low,
            "Cluster 3 should be identified as an ocean cluster with low threshold",
        )
