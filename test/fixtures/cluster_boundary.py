import dataframely as dy
import polars as pl
import shapely

from src.dataframes.cluster_boundary import ClusterBoundarySchema


def mock_cluster_boundary_df() -> dy.DataFrame[ClusterBoundarySchema]:
    """
    Creates a mock ClusterBoundaryDataFrame for testing.
    """
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
    cluster_boundary_df = pl.DataFrame(
        [
            {"cluster": 1, "geometry": shapely.to_wkb(polygon1)},
            {"cluster": 2, "geometry": shapely.to_wkb(polygon2)},
        ]
    ).with_columns(pl.col("cluster").cast(pl.UInt32()))

    return ClusterBoundarySchema.validate(cluster_boundary_df)
