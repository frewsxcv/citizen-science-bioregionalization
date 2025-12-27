import unittest

import polars as pl

from src.darwin_core_utils import (
    build_taxon_filter,
    get_parquet_to_darwin_core_column_mapping,
)


def rename_parquet_columns_to_darwin_core(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Helper function for tests: rename columns from lowercase to camelCase.
    """
    return lf.rename(
        get_parquet_to_darwin_core_column_mapping(),
        strict=False,
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

        lf = df.lazy()

        # Apply the renaming function
        renamed_lf = rename_parquet_columns_to_darwin_core(lf)

        # Collect to check the results
        result_df = renamed_lf.collect()

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

        lf = df.lazy()

        # Apply the renaming function
        renamed_lf = rename_parquet_columns_to_darwin_core(lf)

        # Collect to check the results
        result_df = renamed_lf.collect()

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

    def test_rename_parquet_columns_empty_df(self):
        """Test renaming an empty df"""
        # Create an empty LazyFrame with lowercase column names
        df = pl.DataFrame(
            {
                "decimallatitude": pl.Series([], dtype=pl.Float64),
                "decimallongitude": pl.Series([], dtype=pl.Float64),
            }
        )

        lf = df.lazy()

        # Apply the renaming function
        renamed_lf = rename_parquet_columns_to_darwin_core(lf)

        # Collect to check the results
        result_df = renamed_lf.collect()

        # Check that columns were renamed even though dataframe is empty
        self.assertIn("decimalLatitude", result_df.columns)
        self.assertIn("decimalLongitude", result_df.columns)
        self.assertEqual(len(result_df), 0)


if __name__ == "__main__":
    unittest.main()
