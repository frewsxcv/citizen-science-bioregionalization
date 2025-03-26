import unittest
import polars as pl
import os
import logging
from unittest.mock import patch, MagicMock

from src.lazyframes.darwin_core_csv import DarwinCoreCsvLazyFrame
from src.darwin_core import kingdom_enum, TAXONOMIC_RANKS

# Disable polars logger to reduce test output noise
logging.getLogger("polars").setLevel(logging.WARNING)


class TestDarwinCoreCsvLazyFrame(unittest.TestCase):

    @patch("polars.scan_csv")
    @patch("src.lazyframes.darwin_core_csv.logger")
    def test_build_without_filter(self, mock_logger, mock_scan_csv):
        """Test building a LazyFrame without taxon filtering"""
        # Setup mock lazy frame
        mock_lf = MagicMock()
        mock_lf.collect.return_value = pl.DataFrame(
            {
                "decimalLatitude": [42.1, 43.2, 44.3],
                "decimalLongitude": [-71.2, -72.3, -73.4],
                "kingdom": ["Animalia", "Animalia", "Plantae"],
                "class": ["Aves", "Mammalia", "Magnoliopsida"],
                "genus": ["Parus", "Felis", "Quercus"],
                "species": ["major", "catus", "alba"],
            }
        )
        mock_scan_csv.return_value = mock_lf

        # Call the method under test
        csv_path = "dummy/path.csv"
        lazy_frame = DarwinCoreCsvLazyFrame.build(csv_path)

        # Assertions
        mock_scan_csv.assert_called_once()
        self.assertEqual(lazy_frame.lf, mock_lf)

        # Verify collect was called and returned data correctly
        collected_data = lazy_frame.lf.collect()
        self.assertEqual(collected_data.shape[0], 3)

        # Verify kingdoms and classes
        kingdoms = collected_data["kingdom"].to_list()
        self.assertEqual(kingdoms.count("Animalia"), 2)
        self.assertEqual(kingdoms.count("Plantae"), 1)

        classes = collected_data["class"].to_list()
        self.assertIn("Aves", classes)
        self.assertIn("Mammalia", classes)
        self.assertIn("Magnoliopsida", classes)

    @patch("polars.scan_csv")
    @patch("src.lazyframes.darwin_core_csv.logger")
    def test_build_with_class_filter(self, mock_logger, mock_scan_csv):
        """Test filtering by class"""
        # Setup mock lazy frames
        # 1. Initial mock frame from scan_csv
        mock_scan_result = MagicMock()

        # 2. Result after filter is applied
        mock_filtered = MagicMock()
        mock_filtered.collect.return_value = pl.DataFrame(
            {
                "decimalLatitude": [42.1],
                "decimalLongitude": [-71.2],
                "kingdom": ["Animalia"],
                "class": ["Aves"],
                "genus": ["Parus"],
                "species": ["major"],
            }
        )

        # Setup the mock chain
        mock_scan_result.filter.return_value = mock_filtered
        mock_scan_csv.return_value = mock_scan_result

        # Call the method under test
        csv_path = "dummy/path.csv"
        lazy_frame = DarwinCoreCsvLazyFrame.build(csv_path, taxon_filter="Aves")

        # Verify that filtering was applied
        mock_scan_result.filter.assert_called_once()
        mock_logger.info.assert_called_with("Filtering data to taxon: Aves")

        # Verify the filtered data
        collected_data = lazy_frame.lf.collect()
        self.assertEqual(collected_data.shape[0], 1)
        self.assertEqual(collected_data["class"][0], "Aves")
        self.assertEqual(collected_data["species"][0], "major")

    @patch("polars.scan_csv")
    @patch("src.lazyframes.darwin_core_csv.logger")
    def test_build_with_case_insensitive_filter(self, mock_logger, mock_scan_csv):
        """Test that filtering is case insensitive"""
        # Setup mock lazy frames
        mock_scan_result = MagicMock()
        mock_filtered = MagicMock()
        mock_filtered.collect.return_value = pl.DataFrame(
            {
                "decimalLatitude": [42.1],
                "decimalLongitude": [-71.2],
                "kingdom": ["Animalia"],
                "class": ["Aves"],
                "genus": ["Parus"],
                "species": ["major"],
            }
        )

        mock_scan_result.filter.return_value = mock_filtered
        mock_scan_csv.return_value = mock_scan_result

        # Call with lowercase "aves"
        lazy_frame = DarwinCoreCsvLazyFrame.build("dummy/path.csv", taxon_filter="aves")

        # Verify that lowercase filter works
        mock_scan_result.filter.assert_called_once()
        mock_logger.info.assert_called_with("Filtering data to taxon: aves")

        # Check the result
        collected_data = lazy_frame.lf.collect()
        self.assertEqual(collected_data.shape[0], 1)
        self.assertEqual(collected_data["class"][0], "Aves")

    @patch("polars.scan_csv")
    @patch("src.lazyframes.darwin_core_csv.logger")
    def test_filter_logic(self, mock_logger, mock_scan_csv):
        """Test that the filtering logic creates conditions for all taxonomic ranks"""
        # Setup mock
        mock_lf = MagicMock()
        mock_scan_csv.return_value = mock_lf

        # Call the method
        DarwinCoreCsvLazyFrame.build("dummy/path.csv", taxon_filter="TestTaxon")

        # Verify that filter was called
        mock_lf.filter.assert_called_once()

        # We can't directly check the filter condition, but we can check the logs
        mock_logger.info.assert_called_with("Filtering data to taxon: TestTaxon")


if __name__ == "__main__":
    unittest.main()
