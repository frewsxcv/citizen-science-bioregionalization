"""Tests for the two occurrence-level geocode filters.

Both deliberately filter occurrence records rather than the geocode set, so that
the geocodes and the taxa counts are derived from the same rows.
"""

import unittest

import polars as pl

from src.geocode import filter_sparse_geocodes_lf, filter_terrestrial_geocodes_lf

PRECISION = 5


def _occurrences(points: list[tuple[float, float]]) -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "decimalLatitude": [p[0] for p in points],
            "decimalLongitude": [p[1] for p in points],
            "scientificName": [f"Genus sp{i}" for i in range(len(points))],
        }
    )


class TestFilterSparseGeocodes(unittest.TestCase):
    def test_drops_hexagons_below_the_record_floor(self):
        # Three records in one hexagon, one in another far away.
        lf = _occurrences([(4.0, -74.0), (4.0, -74.0), (4.0, -74.0), (-2.0, -70.0)])

        result = filter_sparse_geocodes_lf(lf, PRECISION, min_records=2).collect()

        self.assertEqual(result.height, 3)

    def test_floor_of_one_keeps_everything(self):
        points = [(4.0, -74.0), (-2.0, -70.0)]
        lf = _occurrences(points)

        result = filter_sparse_geocodes_lf(lf, PRECISION, min_records=1).collect()

        self.assertEqual(result.height, len(points))

    def test_preserves_columns(self):
        lf = _occurrences([(4.0, -74.0), (4.0, -74.0)])

        result = filter_sparse_geocodes_lf(lf, PRECISION, min_records=1)

        self.assertEqual(result.collect_schema().names(), lf.collect_schema().names())


class TestFilterTerrestrialGeocodes(unittest.TestCase):
    def test_keeps_inland_records(self):
        # Bogota, the Llanos, and the Amazon.
        lf = _occurrences([(4.7, -74.1), (4.5, -71.0), (-1.0, -70.0)])

        result = filter_terrestrial_geocodes_lf(lf, PRECISION).collect()

        self.assertEqual(result.height, 3)

    def test_drops_open_ocean_records(self):
        # Open Caribbean and Pacific, well away from any coast.
        lf = _occurrences([(13.0, -76.0), (4.0, -79.5)])

        result = filter_terrestrial_geocodes_lf(lf, PRECISION).collect()

        self.assertEqual(result.height, 0)

    def test_separates_land_from_sea_in_one_frame(self):
        lf = _occurrences([(4.7, -74.1), (13.0, -76.0), (-1.0, -70.0)])

        result = filter_terrestrial_geocodes_lf(lf, PRECISION).collect()

        self.assertEqual(
            sorted(result["decimalLatitude"].to_list()), [-1.0, 4.7]
        )

    def test_preserves_columns(self):
        lf = _occurrences([(4.7, -74.1)])

        result = filter_terrestrial_geocodes_lf(lf, PRECISION)

        self.assertEqual(result.collect_schema().names(), lf.collect_schema().names())


if __name__ == "__main__":
    unittest.main()
