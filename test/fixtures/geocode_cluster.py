import dataframely as dy
import polars as pl

from src.dataframes.geocode_cluster import GeocodeClusterSchema


def mock_geocode_cluster_df() -> dy.DataFrame[GeocodeClusterSchema]:
    """
    Creates a mock GeocodeClusterDataFrame for testing.
    """
    geocode_cluster_data = [
        {"geocode": 1000, "cluster": 0},
        {"geocode": 2000, "cluster": 0},
        {"geocode": 3000, "cluster": 1},
    ]
    df = pl.DataFrame(geocode_cluster_data).with_columns(
        pl.col("geocode").cast(pl.UInt64),
        pl.col("cluster").cast(pl.UInt32),
    )
    return GeocodeClusterSchema.validate(df)
