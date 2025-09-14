import polars as pl
from src.dataframes.cluster_taxa_statistics import ClusterTaxaStatisticsDataFrame


def mock_cluster_taxa_statistics_dataframe() -> ClusterTaxaStatisticsDataFrame:
    """
    Creates a mock ClusterTaxaStatisticsDataFrame for testing.
    """
    taxa_stats_df = pl.DataFrame(
        {
            "cluster": [1, 1, 2, 2, 3, 3, 4, 4, None, None],
            "taxonId": [101, 102, 101, 102, 101, 102, 101, 102, 101, 102],
            "count": [8, 2, 2, 8, 7, 3, 3, 7, 20, 20],
            "average": [0.8, 0.2, 0.2, 0.8, 0.7, 0.3, 0.3, 0.7, 0.5, 0.5],
        },
        schema={
            "cluster": pl.UInt32(),
            "taxonId": pl.UInt32(),
            "count": pl.UInt32(),
            "average": pl.Float64(),
        },
    )
    return ClusterTaxaStatisticsDataFrame(taxa_stats_df)