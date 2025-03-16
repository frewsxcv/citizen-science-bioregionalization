import unittest

import polars as pl
import polars_st
import shapely

from src.geojson import build_geojson_feature_collection
from src.dataframes.cluster_boundary import ClusterBoundaryDataFrame
from src.dataframes.cluster_color import ClusterColorDataFrame


class TestGeojson(unittest.TestCase):
    def test_build_geojson_feature_collection(self):
        # Create mock polygon geometries
        polygon1 = shapely.Polygon([(-67.762704, 49.598604), (-68.147336, 49.56361), (-68.295514, 49.333781), 
                                   (-68.061716, 49.140104), (-67.680942, 49.174866), (-67.530135, 49.403534), 
                                   (-67.762704, 49.598604)])
        polygon2 = shapely.Polygon([(-69.201662, 49.028234), (-69.342511, 48.797701), (-69.719479, 48.757432), 
                                   (-69.959318, 48.947367), (-69.821222, 49.178989), (-70.063974, 49.369372), 
                                   (-69.925042, 49.60178), (-69.540512, 49.642718), (-69.298758, 49.450906), 
                                   (-69.44051, 49.219589), (-69.201662, 49.028234)])
        
        # Create a mock ClusterBoundaryDataFrame
        cluster_boundary_df = pl.DataFrame(
            [
                {"cluster": 1, "boundary": shapely.to_wkb(polygon1)},
                {"cluster": 2, "boundary": shapely.to_wkb(polygon2)},
            ]
        ).with_columns(
            pl.col("cluster").cast(pl.UInt32())
        )
        
        cluster_boundary_dataframe = ClusterBoundaryDataFrame(df=cluster_boundary_df)
        
        actual = build_geojson_feature_collection(
            cluster_boundary_dataframe=cluster_boundary_dataframe,
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
