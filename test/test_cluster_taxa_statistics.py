import unittest
import polars as pl
from src.darwin_core import kingdom_enum
from unittest.mock import patch

from src.dataframes.cluster_taxa_statistics import ClusterTaxaStatisticsDataFrame, assert_dataframe_schema
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
            {"kingdom": "Animalia", "phylum": "Chordata", "class": "Mammalia", "order": "Carnivora",
             "family": "Felidae", "genus": "Panthera", "species": "Panthera leo",
             "taxonRank": "species", "scientificName": "Panthera leo"},
            {"kingdom": "Animalia", "phylum": "Chordata", "class": "Mammalia", "order": "Carnivora",
             "family": "Canidae", "genus": "Canis", "species": "Canis lupus",
             "taxonRank": "species", "scientificName": "Canis lupus"},
            {"kingdom": "Plantae", "phylum": "Tracheophyta", "class": "Magnoliopsida", "order": "Fagales",
             "family": "Fagaceae", "genus": "Quercus", "species": "Quercus robur",
             "taxonRank": "species", "scientificName": "Quercus robur"},
            {"kingdom": "Animalia", "phylum": "Chordata", "class": "Aves", "order": "Anseriformes",
             "family": "", "genus": "", "species": "",
             "taxonRank": "order", "scientificName": "Anseriformes"},
        ]

        # Create a taxonomy DataFrame with proper schema
        taxonomy_df = pl.DataFrame(
            taxonomy_data,
            schema=TaxonomyDataFrame.SCHEMA
        )
        taxonomy_dataframe = TaxonomyDataFrame(taxonomy_df)

        # 2. Create geocode_taxa_counts dataframe
        geocode_taxa_counts_data = [
            {"geocode": "geo1", "kingdom": "Animalia", "taxonRank": "species", "scientificName": "Panthera leo", "count": 5},
            {"geocode": "geo1", "kingdom": "Animalia", "taxonRank": "species", "scientificName": "Canis lupus", "count": 3},
            {"geocode": "geo2", "kingdom": "Animalia", "taxonRank": "species", "scientificName": "Panthera leo", "count": 2},
            {"geocode": "geo2", "kingdom": "Plantae", "taxonRank": "species", "scientificName": "Quercus robur", "count": 8},
            {"geocode": "geo3", "kingdom": "Plantae", "taxonRank": "species", "scientificName": "Quercus robur", "count": 4},
            {"geocode": "geo3", "kingdom": "Animalia", "taxonRank": "order", "scientificName": "Anseriformes", "count": 6},
        ]

        # Create a geocode_taxa_counts DataFrame with proper schema
        geocode_taxa_counts_df = pl.DataFrame(
            geocode_taxa_counts_data,
            schema=GeocodeTaxaCountsDataFrame.SCHEMA
        )
        geocode_taxa_counts_dataframe = GeocodeTaxaCountsDataFrame(geocode_taxa_counts_df)

        # 3. Create geocode_cluster dataframe
        geocode_cluster_data = [
            {"geocode": "geo1", "cluster": 0},
            {"geocode": "geo2", "cluster": 0},
            {"geocode": "geo3", "cluster": 1},
        ]

        # Create a geocode_cluster DataFrame with proper schema
        geocode_cluster_df = pl.DataFrame(
            geocode_cluster_data,
            schema=GeocodeClusterDataFrame.SCHEMA
        )
        geocode_cluster_dataframe = GeocodeClusterDataFrame(geocode_cluster_df)

        # 4. Use our SchemaFixerDataFrame class to test the build method
        # This still tests the original logic but addresses schema validation issues
        result = ClusterTaxaStatisticsDataFrame.build(
            geocode_taxa_counts_dataframe,
            geocode_cluster_dataframe,
            taxonomy_dataframe
        )

        # 5. Verify the result has the expected structure and values

        # Test that resulting dataframe structure is correct
        self.assertIsNotNone(result.df)
        self.assertIn("cluster", result.df.columns)
        self.assertIn("kingdom", result.df.columns)
        self.assertIn("scientificName", result.df.columns)

        # Test the summary row (null cluster) is present
        summary_rows = result.df.filter(pl.col("cluster").is_null())
        self.assertGreater(len(summary_rows), 0)

        # Test cluster specific rows are present
        cluster0_rows = result.df.filter(pl.col("cluster") == 0)
        cluster1_rows = result.df.filter(pl.col("cluster") == 1)
        self.assertGreater(len(cluster0_rows), 0)
        self.assertGreater(len(cluster1_rows), 0)

        # Test that counts are correctly aggregated for Panthera leo (lions)
        leo_in_cluster0 = result.df.filter(
            (pl.col("cluster") == 0) &
            (pl.col("scientificName") == "Panthera leo")
        )
        self.assertEqual(leo_in_cluster0["count"][0], 7)  # 5 from geo1 + 2 from geo2

        # Test that averages are correctly calculated
        # Total observations in cluster 0: 5 + 3 + 2 + 8 = 18
        # So average for Panthera leo should be 7/18 ≈ 0.389
        self.assertAlmostEqual(leo_in_cluster0["average"][0], 7/18, places=3)