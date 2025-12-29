import dataframely as dy
import polars as pl

from src.dataframes.geocode_cluster import GeocodeClusterSchema


def mock_geocode_cluster_df(num_clusters: int = 2) -> dy.DataFrame[GeocodeClusterSchema]:
    """
    Creates a mock GeocodeClusterDataFrame for testing.

    Args:
        num_clusters: Number of clusters to use in the mock data (default: 2)
    """
    geocode_cluster_data = [
        {"geocode": 1000, "num_clusters": num_clusters, "cluster": 0},
        {"geocode": 2000, "num_clusters": num_clusters, "cluster": 0},
        {"geocode": 3000, "num_clusters": num_clusters, "cluster": 1},
    ]
    df = pl.DataFrame(geocode_cluster_data).with_columns(
        pl.col("geocode").cast(pl.UInt64),
        pl.col("num_clusters").cast(pl.UInt32),
        pl.col("cluster").cast(pl.UInt32),
    )
    return GeocodeClusterSchema.validate(df)
