"""Invariants that make a seeded run reproducible.

Two seeded runs on identical input used to return different clusterings. Three
separate causes had to be removed, and each is cheap to assert directly:

1. `taxonId` is a positional index over an unordered `unique()`, so the same
   taxon got a different id each run.
2. The pivot's column order came from another unordered `unique()`.
3. `umap_n_components` defaulted to `n_samples - 2`, whose embedding is not
   reproducible between processes even with a seed.

(1) and (2) matter because UMAP's approximate nearest-neighbour search splits on
feature *indices*, so relabelling or reordering the columns changes the result.
"""

import unittest

import numpy as np
import polars as pl

from src.dataframes.geocode import build_geocode_lf, build_geocode_no_edges_lf
from src.dataframes.darwin_core import build_darwin_core_lf
from src.dataframes.taxonomy import build_taxonomy_lf
from src.matrices.geocode_distance import (
    DEFAULT_UMAP_N_COMPONENTS,
    pivot_taxon_counts,
    reduce_dimensions_umap,
)
from src.types import Bbox

BBOX = Bbox.from_coordinates(-90.0, 90.0, -180.0, 180.0)


class TestPivotColumnOrder(unittest.TestCase):
    def test_columns_are_in_sorted_taxon_id_order(self):
        counts = pl.LazyFrame(
            {
                "geocode": [1, 1, 2, 2],
                "taxonId": [30, 10, 20, 10],
                "count": [1, 2, 3, 4],
            },
            schema={"geocode": pl.UInt64, "taxonId": pl.UInt32, "count": pl.UInt32},
        )

        result = pivot_taxon_counts(counts).collect()

        self.assertEqual(result.columns, ["geocode", "10", "20", "30"])

    def test_column_order_does_not_depend_on_row_order(self):
        rows = {
            "geocode": [1, 1, 2, 2],
            "taxonId": [30, 10, 20, 10],
            "count": [1, 2, 3, 4],
        }
        schema = {"geocode": pl.UInt64, "taxonId": pl.UInt32, "count": pl.UInt32}
        forward = pivot_taxon_counts(pl.LazyFrame(rows, schema=schema)).collect()
        reversed_rows = {k: list(reversed(v)) for k, v in rows.items()}
        backward = pivot_taxon_counts(
            pl.LazyFrame(reversed_rows, schema=schema)
        ).collect()

        self.assertEqual(forward.columns, backward.columns)


class TestTaxonIdAssignment(unittest.TestCase):
    def test_taxon_ids_follow_sorted_names(self):
        occurrences = pl.LazyFrame(
            {
                "decimalLatitude": [4.0, 4.0, 4.0],
                "decimalLongitude": [-74.0, -74.0, -74.0],
                "scientificName": ["Zed zed", "Alpha alpha", "Mid mid"],
                "taxonKey": ["3", "1", "2"],
                "individualCount": [1, 1, 1],
            },
            schema_overrides={"individualCount": pl.Int32},
        )
        geocodes = build_geocode_no_edges_lf(
            build_geocode_lf(occurrences, 4, bounding_box=BBOX)
        )

        result = build_taxonomy_lf(occurrences, 4, geocodes, BBOX).collect()

        self.assertEqual(
            result.sort("taxonId")["scientificName"].to_list(),
            ["Alpha alpha", "Mid mid", "Zed zed"],
        )


class TestUmapDefaults(unittest.TestCase):
    def test_default_component_count_is_small(self):
        # The old default was n_samples - 2, which is not reproducible across
        # processes. Any small fixed value is; 32 is the chosen one.
        self.assertEqual(DEFAULT_UMAP_N_COMPONENTS, 32)

    def test_seeded_reduction_is_repeatable(self):
        rng = np.random.default_rng(0)
        X = pl.from_numpy(rng.random((40, 12)))

        first = reduce_dimensions_umap(X, 4, 0.5, random_state=0).to_numpy()
        second = reduce_dimensions_umap(X, 4, 0.5, random_state=0).to_numpy()

        np.testing.assert_allclose(first, second)


if __name__ == "__main__":
    unittest.main()
