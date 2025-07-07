import unittest
import polars as pl
import os
import tempfile
from src.occurrence_cube_parser import parse_occurrence_cube


class TestOccurrenceCubeParser(unittest.TestCase):
    """
    Unit tests for the parse_occurrence_cube function.
    """
    def test_parse_occurrence_cube(self):
        """
        Tests the parse_occurrence_cube function with dummy data.
        """
        # Use the actual occurrence_cube_sample.csv file from the fixtures directory
        csv_file_path = "test/fixtures/occurrence_cube_sample.csv"

        # Parse the CSV file
        df = parse_occurrence_cube(csv_file_path)

        # Assertions based on the occurrence_cube_sample.csv content
        # Check if the result is a Polars DataFrame
        self.assertIsInstance(df, pl.DataFrame)

        # Check the shape of the DataFrame (number of rows and columns)
        # The header is one line, and there are 39 data rows in occurrence_cube_sample.csv
        expected_rows = 39
        # Read the header line from the actual file to count columns
        with open(csv_file_path, 'r') as f:
            header_line = f.readline()
        expected_cols = len(header_line.strip().split('\t'))
        self.assertEqual(df.shape, (expected_rows, expected_cols))

        # Check if the column names are correct
        expected_column_names = [
            "kingdom", "kingdomkey", "phylum", "phylumkey", "class", "classkey",
            "order", "orderkey", "family", "familykey", "genus", "genuskey",
            "species", "specieskey", "year", "isea3hcellcode", "kingdomcount",
            "phylumcount", "classcount", "ordercount", "familycount", "genuscount",
            "occurrences", "mintemporaluncertainty", "mincoordinateuncertaintyinmeters"
        ]
        self.assertEqual(df.columns, expected_column_names)

        # Optional: Check data types of a few columns
        self.assertEqual(df["year"].dtype, pl.Int64) # Assuming year is integer
        self.assertEqual(df["occurrences"].dtype, pl.Int64) # Assuming occurrences is integer
        self.assertEqual(df["mincoordinateuncertaintyinmeters"].dtype, pl.Float64) # Assuming uncertainty is float


if __name__ == "__main__":
    unittest.main()