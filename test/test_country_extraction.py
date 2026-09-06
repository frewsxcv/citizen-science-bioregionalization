"""Tests for the country-extraction helper scripts.

Only the offline logic is covered: listing and reading a GBIF snapshot needs
network access, so `list_snapshot_files` and `extract_batch` are exercised by
running the script, not by these tests.
"""

import unittest

import polars as pl
import polars_h3

from scripts.extract_country_parquet import (
    SYNTHETIC_TAXON_KEY_BASE,
    assign_dense_taxon_keys,
)
from scripts.filter_sparse_taxa import filter_sparse_taxa_lf

PRECISION = 5


def _occurrences(rows: list[tuple[float, float, str]]) -> pl.LazyFrame:
    """Build an occurrence frame shaped like a GBIF snapshot slice."""
    return pl.LazyFrame(
        {
            "decimallatitude": [r[0] for r in rows],
            "decimallongitude": [r[1] for r in rows],
            "scientificname": [f"Genus {r[2].lower()}" for r in rows],
            "taxonkey": [r[2] for r in rows],
            "individualcount": [1] * len(rows),
        },
        schema_overrides={"individualcount": pl.Int32},
    )


def _hex_count(df: pl.DataFrame) -> int:
    return df.select(
        polars_h3.latlng_to_cell(
            "decimallatitude",
            "decimallongitude",
            resolution=PRECISION,
            return_dtype=pl.UInt64,
        ).n_unique()
    ).item()


class TestAssignDenseTaxonKeys(unittest.TestCase):
    def test_keys_are_offset_out_of_gbif_range(self):
        lf = _occurrences(
            [(4.0, -74.0, "ZZZ"), (2.0, -72.0, "AAA"), (10.0, -75.0, "MMM")]
        )

        remapped, key_map = assign_dense_taxon_keys(lf)
        result = remapped.collect()

        # Every id must sit above the real GBIF backbone range, or Wikidata
        # lookups silently resolve to unrelated taxa.
        self.assertTrue((result["taxonkey"] >= SYNTHETIC_TAXON_KEY_BASE).all())
        self.assertEqual(result["taxonkey"].dtype, pl.UInt32)
        self.assertEqual(
            sorted(key_map["gbif_taxonkey"].to_list()), ["AAA", "MMM", "ZZZ"]
        )

    def test_mapping_is_a_bijection_that_preserves_rows(self):
        lf = _occurrences(
            [
                (4.0, -74.0, "AAA"),
                (4.0, -74.0, "AAA"),
                (2.0, -72.0, "BBB"),
            ]
        )

        remapped, key_map = assign_dense_taxon_keys(lf)
        result = remapped.collect()

        self.assertEqual(result.height, 3)
        self.assertEqual(key_map.height, 2)
        self.assertEqual(result["taxonkey"].n_unique(), 2)
        # Repeated occurrences of one taxon must land on one id.
        self.assertEqual(
            result.filter(pl.col("scientificname") == "Genus aaa")[
                "taxonkey"
            ].n_unique(),
            1,
        )

    def test_output_keeps_the_pipeline_columns(self):
        lf = _occurrences([(4.0, -74.0, "AAA")])

        remapped, _ = assign_dense_taxon_keys(lf)

        self.assertEqual(
            remapped.collect_schema().names(),
            [
                "decimallatitude",
                "decimallongitude",
                "scientificname",
                "taxonkey",
                "individualcount",
            ],
        )


class TestFilterSparseTaxa(unittest.TestCase):
    def test_drops_taxa_below_the_hexagon_threshold(self):
        lf = _occurrences(
            [
                (4.0, -74.0, "WIDE"),
                (2.0, -72.0, "WIDE"),
                (10.0, -75.0, "WIDE"),
                (4.0, -74.0, "RARE"),
            ]
        )

        result = filter_sparse_taxa_lf(lf, PRECISION, min_taxon_hexes=2).collect()

        self.assertEqual(set(result["taxonkey"].to_list()), {"WIDE"})

    def test_hexagons_holding_only_rare_taxa_disappear(self):
        # The whole point of filtering up front: a hexagon emptied by the filter
        # must vanish from the input, so the geocode set and the taxa counts stay
        # in agreement downstream.
        lf = _occurrences(
            [
                (4.0, -74.0, "WIDE"),
                (2.0, -72.0, "WIDE"),
                (-2.0, -70.0, "RARE"),
            ]
        )
        self.assertEqual(_hex_count(lf.collect()), 3)

        result = filter_sparse_taxa_lf(lf, PRECISION, min_taxon_hexes=2).collect()

        self.assertEqual(_hex_count(result), 2)

    def test_threshold_of_one_keeps_everything(self):
        rows = [(4.0, -74.0, "WIDE"), (2.0, -72.0, "RARE")]
        lf = _occurrences(rows)

        result = filter_sparse_taxa_lf(lf, PRECISION, min_taxon_hexes=1).collect()

        self.assertEqual(result.height, len(rows))

    def test_preserves_columns(self):
        lf = _occurrences([(4.0, -74.0, "WIDE"), (2.0, -72.0, "WIDE")])

        result = filter_sparse_taxa_lf(lf, PRECISION, min_taxon_hexes=1)

        self.assertEqual(
            result.collect_schema().names(), lf.collect_schema().names()
        )


if __name__ == "__main__":
    unittest.main()
