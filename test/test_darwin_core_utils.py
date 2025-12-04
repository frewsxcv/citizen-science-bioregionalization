import unittest

import polars as pl

from src.darwin_core_utils import (
    build_taxon_filter,
    get_parquet_to_darwin_core_column_mapping,
    rename_parquet_columns_to_darwin_core,
)


class TestDarwinCoreUtils(unittest.TestCase):
    def test_get_parquet_to_darwin_core_column_mapping(self):
        """Test that the column mapping is returned correctly"""
        mapping = get_parquet_to_darwin_core_column_mapping()

        # Check that it's a dictionary
        self.assertIsInstance(mapping, dict)

        # Check some expected mappings
        self.assertEqual(mapping["decimallatitude"], "decimalLatitude")
        self.assertEqual(mapping["decimallongitude"], "decimalLongitude")
        self.assertEqual(mapping["scientificname"], "scientificName")
        self.assertEqual(mapping["taxonkey"], "taxonKey")

        # Check that all keys are lowercase
        for key in mapping.keys():
            self.assertEqual(key, key.lower())

        # Check that all values use camelCase (start with lowercase)
        for value in mapping.values():
            self.assertTrue(value[0].islower())

    def test_rename_parquet_columns_to_darwin_core(self):
        """Test renaming columns from lowercase to camelCase"""
        # Create a LazyFrame with lowercase column names
        df = pl.DataFrame(
            {
                "decimallatitude": [37.5, 38.0, 39.2],
                "decimallongitude": [-122.1, -122.5, -121.8],
                "scientificname": ["Species A", "Species B", "Species C"],
                "taxonkey": [1, 2, 3],
                "other_column": ["a", "b", "c"],  # column not in mapping
            }
        )

        lazy_frame = df.lazy()

        # Apply the renaming function
        renamed_lazy_frame = rename_parquet_columns_to_darwin_core(lazy_frame)

        # Collect to check the results
        result_df = renamed_lazy_frame.collect()

        # Check that mapped columns were renamed
        self.assertIn("decimalLatitude", result_df.columns)
        self.assertIn("decimalLongitude", result_df.columns)
        self.assertIn("scientificName", result_df.columns)
        self.assertIn("taxonKey", result_df.columns)

        # Check that unmapped columns remain unchanged
        self.assertIn("other_column", result_df.columns)

        # Check that old column names are gone
        self.assertNotIn("decimallatitude", result_df.columns)
        self.assertNotIn("decimallongitude", result_df.columns)
        self.assertNotIn("scientificname", result_df.columns)
        self.assertNotIn("taxonkey", result_df.columns)

        # Verify data integrity
        self.assertEqual(result_df["decimalLatitude"].to_list(), [37.5, 38.0, 39.2])
        self.assertEqual(
            result_df["decimalLongitude"].to_list(), [-122.1, -122.5, -121.8]
        )

    def test_rename_parquet_columns_with_partial_columns(self):
        """Test renaming when only some columns exist in the dataframe"""
        # Create a LazyFrame with only some of the mapped columns
        df = pl.DataFrame(
            {
                "decimallatitude": [37.5, 38.0],
                "decimallongitude": [-122.1, -122.5],
                # scientificname and taxonkey are missing
                "custom_field": ["x", "y"],
            }
        )

        lazy_frame = df.lazy()

        # Apply the renaming function
        renamed_lazy_frame = rename_parquet_columns_to_darwin_core(lazy_frame)

        # Collect to check the results
        result_df = renamed_lazy_frame.collect()

        # Check that only existing mapped columns were renamed
        self.assertIn("decimalLatitude", result_df.columns)
        self.assertIn("decimalLongitude", result_df.columns)

        # Check that non-existent mapped columns weren't added
        self.assertNotIn("scientificName", result_df.columns)
        self.assertNotIn("taxonKey", result_df.columns)

        # Check that unmapped columns remain
        self.assertIn("custom_field", result_df.columns)

        # Should only have 3 columns total
        self.assertEqual(len(result_df.columns), 3)

    def test_rename_parquet_columns_empty_dataframe(self):
        """Test renaming an empty dataframe"""
        # Create an empty LazyFrame with lowercase column names
        df = pl.DataFrame(
            {
                "decimallatitude": pl.Series([], dtype=pl.Float64),
                "decimallongitude": pl.Series([], dtype=pl.Float64),
            }
        )

        lazy_frame = df.lazy()

        # Apply the renaming function
        renamed_lazy_frame = rename_parquet_columns_to_darwin_core(lazy_frame)

        # Collect to check the results
        result_df = renamed_lazy_frame.collect()

        # Check that columns were renamed even though dataframe is empty
        self.assertIn("decimalLatitude", result_df.columns)
        self.assertIn("decimalLongitude", result_df.columns)
        self.assertEqual(len(result_df), 0)

    def test_build_taxon_filter(self):
        """Test building a taxon filter expression"""
        # Create a dataframe with taxonomic data
        df = pl.DataFrame(
            {
                "kingdom": ["Animalia", "Plantae", "Fungi"],
                "phylum": ["Chordata", "Tracheophyta", "Basidiomycota"],
                "class": ["Mammalia", "Magnoliopsida", "Agaricomycetes"],
                "order": ["Carnivora", "Rosales", "Agaricales"],
                "family": ["Felidae", "Rosaceae", "Amanitaceae"],
                "genus": ["Panthera", "Rosa", "Amanita"],
                "species": ["Panthera leo", "Rosa canina", "Amanita muscaria"],
            }
        )

        # Test filtering by kingdom
        filter_expr = build_taxon_filter("Animalia")
        filtered_df = df.filter(filter_expr)
        self.assertEqual(len(filtered_df), 1)
        self.assertEqual(filtered_df["kingdom"][0], "Animalia")

        # Test filtering by phylum
        filter_expr = build_taxon_filter("Chordata")
        filtered_df = df.filter(filter_expr)
        self.assertEqual(len(filtered_df), 1)
        self.assertEqual(filtered_df["phylum"][0], "Chordata")

        # Test filtering by class
        filter_expr = build_taxon_filter("Mammalia")
        filtered_df = df.filter(filter_expr)
        self.assertEqual(len(filtered_df), 1)
        self.assertEqual(filtered_df["class"][0], "Mammalia")

        # Test filtering by genus
        filter_expr = build_taxon_filter("Rosa")
        filtered_df = df.filter(filter_expr)
        self.assertEqual(len(filtered_df), 1)
        self.assertEqual(filtered_df["genus"][0], "Rosa")

    def test_build_taxon_filter_no_matches(self):
        """Test that taxon filter returns no results when there's no match"""
        df = pl.DataFrame(
            {
                "kingdom": ["Animalia", "Plantae"],
                "phylum": ["Chordata", "Tracheophyta"],
                "class": ["Mammalia", "Magnoliopsida"],
                "order": ["Carnivora", "Rosales"],
                "family": ["Felidae", "Rosaceae"],
                "genus": ["Panthera", "Rosa"],
                "species": ["Panthera leo", "Rosa canina"],
            }
        )

        # Test filtering by a taxon that doesn't exist
        filter_expr = build_taxon_filter("Fungi")
        filtered_df = df.filter(filter_expr)
        self.assertEqual(len(filtered_df), 0)

    def test_build_taxon_filter_multiple_matches(self):
        """Test that taxon filter can match across different taxonomic ranks"""
        df = pl.DataFrame(
            {
                "kingdom": ["Animalia", "Animalia", "Plantae"],
                "phylum": ["Chordata", "Arthropoda", "Tracheophyta"],
                "class": ["Mammalia", "Insecta", "Magnoliopsida"],
                "order": ["Carnivora", "Lepidoptera", "Rosales"],
                "family": ["Felidae", "Nymphalidae", "Rosaceae"],
                "genus": ["Panthera", "Vanessa", "Rosa"],
                "species": ["Panthera leo", "Vanessa atalanta", "Rosa canina"],
            }
        )

        # Test filtering by kingdom that appears multiple times
        filter_expr = build_taxon_filter("Animalia")
        filtered_df = df.filter(filter_expr)
        self.assertEqual(len(filtered_df), 2)

        # All filtered rows should have "Animalia" as kingdom
        self.assertTrue(all(filtered_df["kingdom"] == "Animalia"))


if __name__ == "__main__":
    unittest.main()
