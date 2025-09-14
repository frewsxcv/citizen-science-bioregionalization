import polars as pl
from src.dataframes.geocode_cluster import GeocodeClusterDataFrame


def mock_geocode_cluster_dataframe() -> GeocodeClusterDataFrame:
    """
    Creates a mock GeocodeClusterDataFrame for testing.
    """
    geocode_cluster_data = [
        {"geocode": 1000, "cluster": 0},
        {"geocode": 2000, "cluster": 0},
        {"geocode": 3000, "cluster": 1},
    ]
    geocode_cluster_df = pl.DataFrame(
        geocode_cluster_data, schema=GeocodeClusterDataFrame.SCHEMA
    )
    return GeocodeClusterDataFrame(geocode_cluster_df)