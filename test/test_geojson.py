import unittest

import polars as pl

from src.geojson import build_geojson_feature_collection
from src.geojson import GeocodeClusterDataFrame, ClusterColorDataFrame


class TestGeojson(unittest.TestCase):
    def test_build_geojson_feature_collection(self):
        actual = build_geojson_feature_collection(
            geocode_cluster_dataframe=GeocodeClusterDataFrame(
                df=pl.DataFrame(
                    [
                        {"geocode": "840e491ffffffff", "cluster": 1},
                        {"geocode": "840e4b1ffffffff", "cluster": 2},
                        {"geocode": "840e4b3ffffffff", "cluster": 2},
                    ]
                ),
            ),
            cluster_colors_dataframe=ClusterColorDataFrame(
                df=pl.DataFrame(
                    [
                        {"cluster": 1, "color": "#ff0000"},
                        {"cluster": 2, "color": "#0000ff"},
                    ],
                    schema=ClusterColorDataFrame.SCHEMA,
                )
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
                    "properties": {"cluster": 1, "fill": "#ff0000", "stroke-width": 0},
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
                    "properties": {"cluster": 2, "fill": "#0000ff", "stroke-width": 0},
                    "type": "Feature",
                },
            ],
            "type": "FeatureCollection",
        }
        self.assertEqual(expected, actual)
