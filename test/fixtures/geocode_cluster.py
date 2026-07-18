import polars as pl



def mock_geocode_cluster_df(num_clusters: int = 2) -> pl.DataFrame:
    """
    Creates a mock GeocodeClusterDataFrame for testing (single k value).

    Args:
        num_clusters: Number of clusters to use in the mock data (default: 2)
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
    return df


def mock_geocode_cluster_multi_k_df(
    num_clusters: int = 2,
) -> pl.DataFrame:
    """
    Creates a mock GeocodeClusterDataFrame for testing (with num_clusters field).

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
    return df
