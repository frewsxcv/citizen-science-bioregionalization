import geojson
import unittest

from src.cluster_stats import Stats
from src.geojson import build_geojson_feature_collection
from src.geohash import Geohash


class TestGeojson(unittest.TestCase):
    def test_build_geojson_feature_collection(self):
        cluster_and_geohashes_and_colors = [
            (1, ["gcpv"], "red"),
            (2, ["gcpv", "gcpw"], "blue"),
        ]
        actual = build_geojson_feature_collection(
            cluster_and_geohashes_and_colors
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
