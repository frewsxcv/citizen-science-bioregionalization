import unittest

import polars as pl

from src.geojson import build_geojson_feature_collection
from src.geojson import GeohashClusterDataFrame, ClusterColorDataFrame


class TestGeojson(unittest.TestCase):
    def test_build_geojson_feature_collection(self):
        actual = build_geojson_feature_collection(
            geohash_cluster_dataframe=GeohashClusterDataFrame(
                df=pl.DataFrame(
                    [
                        {"geohash": "gcpv", "cluster": 1},
                        {"geohash": "gcpw", "cluster": 2},
                        {"geohash": "gcpx", "cluster": 2},
                    ]
                ),
            ),
            cluster_colors_dataframe=ClusterColorDataFrame(
                df=pl.DataFrame(
                    [
                        {"cluster": 1, "color": "#ff0000"},
                        {"cluster": 2, "color": "#0000ff"},
                    ]
                )
            ),
        )
        expected = {
            "features": [
                {
                    "geometry": {
                        "coordinates": [
                            [
                                [-0.351562, 51.503906],
                                [0.0, 51.503906],
                                [0.0, 51.679688],
                                [-0.351562, 51.679688],
                                [-0.351562, 51.503906],
                            ]
                        ],
                        "type": "Polygon",
                    },
                    "properties": {"cluster": 1, "fill": "red", "stroke-width": 0},
                    "type": "Feature",
                },
                {
                    "geometry": {
                        "coordinates": [
                            [
                                [
                                    [0.0, 51.679688],
                                    [0.0, 51.503906],
                                    [-0.351562, 51.503906],
                                    [-0.351562, 51.679688],
                                    [0.0, 51.679688],
                                ]
                            ],
                            [
                                [
                                    [-0.703125, 51.679688],
                                    [-0.703125, 51.855469],
                                    [-0.351562, 51.855469],
                                    [-0.351562, 51.679688],
                                    [-0.703125, 51.679688],
                                ]
                            ],
                        ],
                        "type": "MultiPolygon",
                    },
                    "properties": {"cluster": 2, "fill": "blue", "stroke-width": 0},
                    "type": "Feature",
                },
            ],
            "type": "FeatureCollection",
        }
        self.assertEqual(expected, actual)
