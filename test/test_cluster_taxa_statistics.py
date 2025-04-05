import unittest
import polars as pl
from src.darwin_core import kingdom_enum
from unittest.mock import patch

from src.dataframes.cluster_taxa_statistics import (
    ClusterTaxaStatisticsDataFrame,
    assert_dataframe_schema,
)
from src.dataframes.geocode_cluster import GeocodeClusterDataFrame
from src.dataframes.geocode_taxa_counts import GeocodeTaxaCountsDataFrame
from src.dataframes.taxonomy import TaxonomyDataFrame


class TestClusterTaxaStatistics(unittest.TestCase):
    def test_build(self):
        """
        Test the build method of ClusterTaxaStatisticsDataFrame.

        This test creates input dataframes that match the expected schema
        and verifies that the build method correctly processes them.
        """
        # 1. Create taxonomy dataframe
        taxonomy_data = [
            {
                "taxonId": 0,
                "kingdom": "Animalia",
                "phylum": "Chordata",
                "class": "Mammalia",
                "order": "Carnivora",
                "family": "Felidae",
                "genus": "Panthera",
                "species": "leo",
                "taxonRank": "species",
                "scientificName": "Panthera leo",
            },
            {
                "taxonId": 1,
                "kingdom": "Animalia",
                "phylum": "Chordata",
                "class": "Mammalia",
                "order": "Carnivora",
                "family": "Canidae",
                "genus": "Canis",
                "species": "lupus",
                "taxonRank": "species",
                "scientificName": "Canis lupus",
            },
            {
                "taxonId": 2,
                "kingdom": "Plantae",
                "phylum": "Tracheophyta",
                "class": "Magnoliopsida",
                "order": "Fagales",
                "family": "Fagaceae",
                "genus": "Quercus",
                "species": "robur",
                "taxonRank": "species",
                "scientificName": "Quercus robur",
            },
            {
                "taxonId": 3,
                "kingdom": "Animalia",
                "phylum": "Chordata",
                "class": "Aves",
                "order": "Anseriformes",
                "family": "",
                "genus": "",
                "species": "",
                "taxonRank": "order",
                "scientificName": "Anseriformes",
            },
        ]

        # Create a taxonomy DataFrame with proper schema
        taxonomy_df = pl.DataFrame(taxonomy_data, schema=TaxonomyDataFrame.SCHEMA)
        taxonomy_dataframe = TaxonomyDataFrame(taxonomy_df)

        # 2. Create geocode_taxa_counts dataframe
        geocode_taxa_counts_data = [
            {"geocode": 1000, "taxonId": 0, "count": 5},  # Panthera leo
            {"geocode": 1000, "taxonId": 1, "count": 3},  # Canis lupus
            {"geocode": 2000, "taxonId": 0, "count": 2},  # Panthera leo
            {"geocode": 2000, "taxonId": 2, "count": 8},  # Quercus robur
            {"geocode": 3000, "taxonId": 2, "count": 4},  # Quercus robur
            {"geocode": 3000, "taxonId": 3, "count": 6},  # Anseriformes
        ]

        # Create a geocode_taxa_counts DataFrame with proper schema
        geocode_taxa_counts_df = pl.DataFrame(
            geocode_taxa_counts_data, schema=GeocodeTaxaCountsDataFrame.SCHEMA
        )
        geocode_taxa_counts_dataframe = GeocodeTaxaCountsDataFrame(
            geocode_taxa_counts_df
        )

        # 3. Create geocode_cluster dataframe
        geocode_cluster_data = [
            {"geocode": 1000, "cluster": 0},
            {"geocode": 2000, "cluster": 0},
            {"geocode": 3000, "cluster": 1},
        ]

        # Create a geocode_cluster DataFrame with proper schema
        geocode_cluster_df = pl.DataFrame(
            geocode_cluster_data, schema=GeocodeClusterDataFrame.SCHEMA
        )
        geocode_cluster_dataframe = GeocodeClusterDataFrame(geocode_cluster_df)

        # 4. Use our SchemaFixerDataFrame class to test the build method
        # This still tests the original logic but addresses schema validation issues
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
