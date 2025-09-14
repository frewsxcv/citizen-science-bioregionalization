import polars as pl
from src.dataframes.cluster_color import ClusterColorDataFrame


def mock_cluster_color_dataframe() -> ClusterColorDataFrame:
    """
    Creates a mock ClusterColorDataFrame for testing.
    """
    return ClusterColorDataFrame(
        df=pl.DataFrame(
            [
                {"cluster": 1, "color": "#ff0000", "darkened_color": "#800000"},
                {"cluster": 2, "color": "#0000ff", "darkened_color": "#000080"},
            ],
            schema=ClusterColorDataFrame.SCHEMA,
        )
    )