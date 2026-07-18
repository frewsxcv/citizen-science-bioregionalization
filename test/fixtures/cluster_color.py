import polars as pl



def mock_cluster_color_df() -> pl.DataFrame:
    """
    Creates a mock ClusterColorDataFrame for testing.
    """
    df = pl.DataFrame(
        [
            {"cluster": 1, "color": "#ff0000", "darkened_color": "#800000"},
            {"cluster": 2, "color": "#0000ff", "darkened_color": "#000080"},
        ],
    ).with_columns(pl.col("cluster").cast(pl.UInt32))
    return df
