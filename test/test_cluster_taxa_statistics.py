import unittest
import polars as pl
from polars_darwin_core.darwin_core import kingdom_data_type
from unittest.mock import patch

from src.dataframes.cluster_taxa_statistics import (
    ClusterTaxaStatisticsDataFrame,
    assert_dataframe_schema,
)
from test.fixtures.taxonomy import mock_taxonomy_dataframe
from test.fixtures.geocode_taxa_counts import mock_geocode_taxa_counts_dataframe
from test.fixtures.geocode_cluster import mock_geocode_cluster_dataframe


class TestClusterTaxaStatistics(unittest.TestCase):
    def test_build(self):
        """
        Test the build method of ClusterTaxaStatisticsDataFrame.

        This test creates input dataframes that match the expected schema
        and verifies that the build method correctly processes them.
        """
        taxonomy_dataframe = mock_taxonomy_dataframe()
        geocode_taxa_counts_dataframe = mock_geocode_taxa_counts_dataframe()
        geocode_cluster_dataframe = mock_geocode_cluster_dataframe()

        result = ClusterTaxaStatisticsDataFrame.build(
            geocode_taxa_counts_dataframe, geocode_cluster_dataframe, taxonomy_dataframe
        )

        # 5. Verify the result has the expected structure and values

        # Test that resulting dataframe structure is correct
        self.assertIsNotNone(result.df)
        self.assertIn("cluster", result.df.columns)
        self.assertIn("taxonId", result.df.columns)
        self.assertIn("count", result.df.columns)
        self.assertIn("average", result.df.columns)

        # Test the summary row (null cluster) is present
        summary_rows = result.df.filter(pl.col("cluster").is_null())
        self.assertGreater(len(summary_rows), 0)

        # Test cluster specific rows are present
        cluster0_rows = result.df.filter(pl.col("cluster") == 0)
        cluster1_rows = result.df.filter(pl.col("cluster") == 1)
        self.assertGreater(len(cluster0_rows), 0)
        self.assertGreater(len(cluster1_rows), 0)

        # Test that counts are correctly aggregated for Panthera leo (taxonId 0)
        leo_in_cluster0 = result.df.filter(
            (pl.col("cluster") == 0) & (pl.col("taxonId") == 0)
        )
        self.assertEqual(leo_in_cluster0["count"][0], 7)  # 5 from geo1 + 2 from geo2

        # Test that averages are correctly calculated
        # Total observations in cluster 0: 5 + 3 + 2 + 8 = 18
        # So average for Panthera leo should be 7/18 ≈ 0.389
        self.assertAlmostEqual(leo_in_cluster0["average"][0], 7 / 18, places=3)
