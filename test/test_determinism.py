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
from scipy.spatial.distance import pdist

from src.dataframes.darwin_core import build_darwin_core_lf
from src.dataframes.geocode import build_geocode_lf, build_geocode_no_edges_lf
from src.dataframes.geocode_taxa_counts import build_geocode_taxa_counts_lf
from src.dataframes.taxonomy import build_taxonomy_lf
from src.types import Bbox

from src.dataframes.geocode import build_geocode_lf, build_geocode_no_edges_lf
from src.dataframes.darwin_core import build_darwin_core_lf
from src.dataframes.taxonomy import build_taxonomy_lf
from src.matrices.geocode_distance import (
    pivot_taxon_counts,
    GeocodeDistanceMatrix,
    build_X,
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


class TestDistanceStage(unittest.TestCase):
    def test_distances_are_bray_curtis_on_the_counts(self):
        """No embedding sits between the counts and the distances any more.

        UMAP used to, and it was the pipeline's only nondeterministic stage:
        five CI runs on a byte-identical input matrix produced two different
        embeddings and both k=2 and k=3, on the same runner image.
        """
        rng = np.random.default_rng(0)
        geocodes = [f"8a{i:010d}" for i in range(30)]
        rows = [
            {"geocode": g, "taxonId": int(t), "count": int(c)}
            for g in geocodes
            for t, c in enumerate(rng.integers(0, 20, 25))
            if c > 0
        ]
        counts = pl.DataFrame(rows).with_columns(
            pl.col("taxonId").cast(pl.UInt32), pl.col("count").cast(pl.UInt32)
        )
        present = counts.select("geocode").unique().sort("geocode")

        first = GeocodeDistanceMatrix.build(counts.lazy(), present.lazy())
        second = GeocodeDistanceMatrix.build(counts.lazy(), present.lazy())

        np.testing.assert_array_equal(first.condensed(), second.condensed())
        np.testing.assert_array_equal(
            first.condensed(), pdist(first.features(), metric="braycurtis")
        )


