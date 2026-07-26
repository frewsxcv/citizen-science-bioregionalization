import unittest

import numpy as np
import polars as pl
from scipy.spatial.distance import pdist

from src.matrices.geocode_distance import (
    GeocodeDistanceMatrix,
    binarize_values,
    build_X,
    pivot_taxon_counts,
    scale_values,
)


def taxa_counts_lf(rows: list[tuple[str, int, int]]) -> pl.LazyFrame:
    """Build a (geocode, taxonId, count) frame in the pipeline's schema."""
    return pl.LazyFrame(
        {
            "geocode": [r[0] for r in rows],
            "taxonId": [r[1] for r in rows],
            "count": [r[2] for r in rows],
        },
        schema={"geocode": pl.String, "taxonId": pl.UInt32, "count": pl.UInt32},
    )


def geocode_lf(geocodes: list[str]) -> pl.LazyFrame:
    return pl.LazyFrame({"geocode": geocodes}, schema={"geocode": pl.String})


class TestPivotTaxonCounts(unittest.TestCase):
    def test_columns_are_in_sorted_taxon_order(self):
        """An unstable column order makes a seeded UMAP fit irreproducible."""
        counts = taxa_counts_lf(
            [("a", 71, 1), ("a", 4, 1), ("b", 900, 2), ("b", 12, 1), ("c", 4, 3)]
        )
        columns = pivot_taxon_counts(counts).collect().columns
        self.assertEqual(columns, ["geocode", "4", "12", "71", "900"])

    def test_repeated_pivots_agree(self):
        counts = taxa_counts_lf([("a", i, i + 1) for i in range(40)])
        first = pivot_taxon_counts(counts).collect().columns
        for _ in range(4):
            self.assertEqual(pivot_taxon_counts(counts).collect().columns, first)


class TestBinarizeValues(unittest.TestCase):
    def test_any_positive_count_becomes_one(self):
        result = binarize_values(
            pl.DataFrame({"a": [0, 1, 7, 1000], "b": [3, 0, 0, 2]})
        )
        self.assertEqual(result["a"].to_list(), [0.0, 1.0, 1.0, 1.0])
        self.assertEqual(result["b"].to_list(), [1.0, 0.0, 0.0, 1.0])

    def test_output_is_float(self):
        # UMAP and scipy both accept floats; booleans would need casting at
        # every call site.
        result = binarize_values(pl.DataFrame({"a": [0, 5]}))
        self.assertEqual(result["a"].dtype, pl.Float64)

    def test_shape_preserved(self):
        df = pl.DataFrame({"a": [1, 2, 3], "b": [0, 0, 4], "c": [9, 0, 0]})
        self.assertEqual(binarize_values(df).shape, df.shape)

    def test_is_fit_free(self):
        """The transform must not depend on the other rows in the frame.

        This is the property that makes separately-run facets comparable, and
        the one RobustScaler lacks.
        """
        full = pl.DataFrame({"a": [1, 50, 900], "b": [0, 2, 0]})
        subset = full.head(2)

        self.assertEqual(
            binarize_values(full).head(2).to_dicts(),
            binarize_values(subset).to_dicts(),
        )
        # Contrast: the abundance path's scaler shifts under the same slice.
        self.assertNotEqual(
            scale_values(full).head(2).to_dicts(),
            scale_values(subset).to_dicts(),
        )


class TestBuildX(unittest.TestCase):
    def setUp(self):
        # Three geocodes, three taxa. "a" and "b" share taxon 1; "c" is
        # disjoint from both.
        self.counts = taxa_counts_lf(
            [
                ("a", 1, 5),
                ("a", 2, 1),
                ("b", 1, 40),
                ("c", 3, 2),
            ]
        )
        self.geocodes = geocode_lf(["a", "b", "c"])

    def test_presence_output_is_binary(self):
        X = build_X(self.counts, self.geocodes, metric="presence")
        values = set(X.to_numpy().flatten().tolist())
        self.assertEqual(values, {0.0, 1.0})

    def test_presence_ignores_count_magnitude(self):
        """Geocodes with the same taxa differ only where taxa differ."""
        X = build_X(self.counts, self.geocodes, metric="presence").to_numpy()
        # Row "b" holds taxon 1 at count 40 vs. "a" at 5, but presence is 1 for
        # both, so they agree on that column.
        self.assertTrue(np.array_equal(X[0] > 0, np.array([True, True, False])))
        self.assertTrue(np.array_equal(X[1] > 0, np.array([True, False, False])))
        self.assertTrue(np.array_equal(X[2] > 0, np.array([False, False, True])))

    def test_presence_yields_nonnegative_matrix(self):
        """Bray-Curtis and Sorensen are undefined for negative values.

        The abundance path violates this — RobustScaler centers on the median —
        which is the defect the presence path exists to avoid.
        """
        presence = build_X(self.counts, self.geocodes, metric="presence").to_numpy()
        self.assertTrue((presence >= 0).all())

        abundance = build_X(self.counts, self.geocodes, metric="abundance").to_numpy()
        self.assertTrue((abundance < 0).any())

    def test_defaults_to_abundance(self):
        self.assertTrue(
            np.allclose(
                build_X(self.counts, self.geocodes).to_numpy(),
                build_X(self.counts, self.geocodes, metric="abundance").to_numpy(),
            )
        )

    def test_geocode_order_mismatch_is_rejected(self):
        with self.assertRaises(AssertionError):
            build_X(self.counts, geocode_lf(["c", "b", "a"]), metric="presence")


class TestSorensenDistance(unittest.TestCase):
    """scipy's `dice` metric is the Sorensen dissimilarity on binary vectors."""

    def test_matches_sorensen_formula(self):
        # |A| = 3, |B| = 3, |A n B| = 2  ->  1 - 2*2/(3+3) = 1/3
        a = np.array([1.0, 1.0, 1.0, 0.0])
        b = np.array([1.0, 1.0, 0.0, 1.0])
        self.assertAlmostEqual(float(pdist([a, b], metric="dice")[0]), 1 / 3)

    def test_identical_vectors_are_zero_distance(self):
        a = np.array([1.0, 0.0, 1.0])
        self.assertAlmostEqual(float(pdist([a, a], metric="dice")[0]), 0.0)

    def test_disjoint_vectors_are_unit_distance(self):
        a = np.array([1.0, 1.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 1.0, 1.0])
        self.assertAlmostEqual(float(pdist([a, b], metric="dice")[0]), 1.0)


class TestGeocodeDistanceMatrixBuild(unittest.TestCase):
    """End-to-end build. UMAP is seeded so these are reproducible."""

    def setUp(self):
        # Two compositional groups: geocodes 0-4 share one taxon pool, 5-9
        # another, with a little overlap so the matrix is not degenerate.
        rows: list[tuple[str, int, int]] = []
        for i in range(10):
            pool = range(0, 6) if i < 5 else range(4, 10)
            for taxon in pool:
                rows.append((f"g{i:02d}", taxon, (i % 3) + 1))
        self.counts = taxa_counts_lf(rows)
        self.geocodes = geocode_lf([f"g{i:02d}" for i in range(10)])

    def build(self, metric):
        return GeocodeDistanceMatrix.build(
            self.counts,
            self.geocodes,
            umap_n_components=2,
            metric=metric,
            random_state=17,
        )

    def test_presence_distances_are_valid(self):
        condensed = self.build("presence").condensed()
        self.assertEqual(condensed.shape, (45,))  # 10 choose 2
        self.assertTrue(np.isfinite(condensed).all())
        self.assertTrue((condensed >= 0).all())

    def test_squareform_is_symmetric_with_zero_diagonal(self):
        square = self.build("presence").squareform()
        self.assertEqual(square.shape, (10, 10))
        self.assertTrue(np.allclose(square, square.T))
        self.assertTrue(np.allclose(np.diag(square), 0.0))

    def test_seeded_builds_are_reproducible(self):
        # Required for the null model: without this, run-to-run UMAP noise is
        # indistinguishable from the effect being measured.
        self.assertTrue(
            np.allclose(
                self.build("presence").condensed(),
                self.build("presence").condensed(),
            )
        )

    def test_metrics_produce_different_distances(self):
        self.assertFalse(
            np.allclose(
                self.build("presence").condensed(),
                self.build("abundance").condensed(),
            )
        )

    def test_reduced_features_shape(self):
        self.assertEqual(self.build("presence").reduced_features().shape, (10, 2))


class TestDisjointGeocodes(unittest.TestCase):
    """Geocodes sharing no taxa sit at Sorensen distance exactly 1.0.

    UMAP defaults `disconnection_distance` to 1.0 for dice, so left alone it
    severs every one of those edges, fully disconnects the affected geocodes,
    and returns NaN coordinates for them — which propagate through pdist and
    Ward into the cluster metrics without raising anything.
    """

    def setUp(self):
        # Sparse and lopsided, like real occurrence data at coarse resolution:
        # one well-sampled geocode plus a tail of geocodes holding a single
        # taxon each, disjoint from everything else.
        rows: list[tuple[str, int, int]] = [("g00", t, 1) for t in range(200)]
        for i in range(1, 12):
            rows.append((f"g{i:02d}", 500 + i, 1))
        self.counts = taxa_counts_lf(rows)
        self.geocodes = geocode_lf([f"g{i:02d}" for i in range(12)])

    def test_no_nan_distances(self):
        matrix = GeocodeDistanceMatrix.build(
            self.counts,
            self.geocodes,
            umap_n_components=2,
            metric="presence",
            random_state=3,
        )
        self.assertFalse(np.isnan(matrix.reduced_features()).any())
        self.assertFalse(np.isnan(matrix.condensed()).any())
        self.assertTrue(np.isfinite(matrix.condensed()).all())


if __name__ == "__main__":
    unittest.main()
