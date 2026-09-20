import unittest

import numpy as np
import polars as pl
from scipy.spatial.distance import pdist

import src.matrices.geocode_distance as geocode_distance
from src.matrices.geocode_distance import build_unscaled_X, GeocodeDistanceMatrix


class TestEmbeddingDistanceIsEuclidean(unittest.TestCase):
    """The distance matrix must be Euclidean on the embedding it stores.

    The reduction used to measure the embedding with Bray-Curtis, justified as
    being "consistent with the UMAP metric". That has the relationship
    backwards: UMAP's `metric` describes the input space, and the embedding it
    returns is Euclidean by construction, holding signed coordinates that are
    not abundances -- 27.6% of entries are negative on the sample archive.
    Bray-Curtis, a ratio of summed absolute differences to summed absolute
    totals, says nothing about such a vector. Ward's linkage also assumes
    squared Euclidean geometry, and is handed this matrix directly.
    """

    def _build(self) -> GeocodeDistanceMatrix:
        rng = np.random.default_rng(0)
        geocodes = [f"8a{i:010d}" for i in range(40)]
        rows = [
            {"geocode": g, "taxonId": int(t), "count": int(c)}
            for g in geocodes
            for t, c in enumerate(rng.integers(0, 20, 30))
            if c > 0
        ]
        counts = pl.DataFrame(rows).with_columns(
            pl.col("taxonId").cast(pl.UInt32), pl.col("count").cast(pl.UInt32)
        )
        present = counts.select("geocode").unique().sort("geocode")
        return GeocodeDistanceMatrix.build(
            counts.lazy(), present.lazy(), random_state=0
        )

    def test_condensed_distances_match_euclidean_on_the_embedding(self):
        matrix = self._build()
        np.testing.assert_allclose(
            matrix.condensed(),
            pdist(matrix.reduced_features(), metric="euclidean"),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_distances_are_not_bounded_by_one(self):
        """A weaker but independent signal that the metric is not Bray-Curtis.

        Bray-Curtis is bounded on [0, 1] and compressed the sample archive's
        pairs into 0.0085..0.0531, where Euclidean on the same embedding spanned
        0.62..3.36. Anything confined below 1 would be suspicious.
        """
        matrix = self._build()
        self.assertGreater(matrix.condensed().max(), 1.0)


if __name__ == "__main__":
    unittest.main()


class TestAbundanceReferenceDistances(unittest.TestCase):
    """`build` must also carry distances measured on the abundances themselves.

    These are never clustered on. They exist so the reported silhouette, which
    is measured on the UMAP embedding, can be checked against one measured in
    the space the data actually occupies. Across six configurations the two
    disagreed by 3.4x to 24x, and twice the abundance-space figure was negative
    while the reported one was positive.
    """

    def _counts(self):
        rng = np.random.default_rng(0)
        geocodes = [f"8a{i:010d}" for i in range(40)]
        rows = [
            {"geocode": g, "taxonId": int(t), "count": int(c)}
            for g in geocodes
            for t, c in enumerate(rng.integers(0, 20, 30))
            if c > 0
        ]
        counts = pl.DataFrame(rows).with_columns(
            pl.col("taxonId").cast(pl.UInt32), pl.col("count").cast(pl.UInt32)
        )
        return counts, counts.select("geocode").unique().sort("geocode")

    def test_build_populates_abundance_distances(self):
        counts, present = self._counts()
        matrix = GeocodeDistanceMatrix.build(
            counts.lazy(), present.lazy(), random_state=0
        )
        abundance = matrix.abundance_condensed()
        self.assertIsNotNone(abundance)
        assert abundance is not None
        self.assertEqual(abundance.shape, matrix.condensed().shape)

    def test_abundance_distances_are_bray_curtis_and_bounded(self):
        """Bray-Curtis on non-negative counts is bounded on [0, 1], unlike the
        Euclidean distances the clustering uses."""
        counts, present = self._counts()
        matrix = GeocodeDistanceMatrix.build(
            counts.lazy(), present.lazy(), random_state=0
        )
        abundance = matrix.abundance_condensed()
        assert abundance is not None
        self.assertGreaterEqual(abundance.min(), 0.0)
        self.assertLessEqual(abundance.max(), 1.0)
        self.assertGreater(matrix.condensed().max(), 1.0)

    def test_direct_construction_leaves_them_absent(self):
        """Test fixtures build the matrix directly; that must stay possible."""
        matrix = GeocodeDistanceMatrix(np.array([1.0]), np.zeros((2, 2)))
        self.assertIsNone(matrix.abundance_condensed())


class TestCompositionMetric(unittest.TestCase):
    """Presence reduces each count to whether the taxon was seen at all.

    Counts are not trustworthy here: individualCount has a median of 2 and a
    maximum of 35,182,100, and 19% of records carry none and are filled with 1.
    Measured at k=4 against Bray-Curtis over raw counts, presence raised
    explained variance from 0.0351 to 0.0591 on Colombia and 0.0468 to 0.1114 on
    southeast Australia, with the best separation of seven representations
    tried.
    """

    def _counts(self):
        rng = np.random.default_rng(0)
        geocodes = [f"8a{i:010d}" for i in range(30)]
        rows = []
        for g in geocodes:
            for t, c in enumerate(rng.integers(1, 30, 25)):
                rows.append({"geocode": g, "taxonId": int(t), "count": int(c)})
        # One cell carries an absurd count, as the real data does: individualCount
        # has a median of 2 and a maximum of 35,182,100.
        rows[0]["count"] = 35_000_000
        counts = pl.DataFrame(rows).with_columns(
            pl.col("taxonId").cast(pl.UInt32), pl.col("count").cast(pl.UInt32)
        )
        return counts, counts.select("geocode").unique().sort("geocode")

    def _reference(self, counts, present, metric):
        """The pre-UMAP distances, which are taken over the composition vectors
        and so reveal what the metric did without storing the matrix itself."""
        matrix = GeocodeDistanceMatrix.build(
            counts.lazy(), present.lazy(), metric=metric
        )
        ref = matrix.abundance_condensed()
        assert ref is not None
        return ref

    def test_presence_measures_sorensen_over_bits(self):
        """Bray-Curtis over presence bits is Sorensen, so the reference
        distances must equal those from an explicitly binarised matrix."""
        counts, present = self._counts()
        wide = (
            counts.pivot(on="taxonId", index="geocode", values="count")
            .sort("geocode")
            .drop("geocode")
            .fill_null(0)
        )
        binary = (wide.to_numpy() > 0).astype(np.float64)
        np.testing.assert_allclose(
            self._reference(counts, present, "presence"),
            pdist(binary, metric="braycurtis"),
            rtol=1e-9,
            atol=1e-9,
        )

    def test_presence_distances_stay_within_the_bound(self):
        """RobustScaler on a binary column whose median is 1 maps it to 0 and
        -1, handing Bray-Curtis the negatives it is not defined for. Presence
        skips scaling, so the distances stay on [0, 1]."""
        counts, present = self._counts()
        ref = self._reference(counts, present, "presence")
        self.assertGreaterEqual(ref.min(), 0.0)
        self.assertLessEqual(ref.max(), 1.0)

    def test_presence_ignores_an_absurd_count(self):
        """A single nine-million record must not move a presence vector at all."""
        counts, present = self._counts()
        inflated = counts.with_columns(
            pl.when(pl.int_range(pl.len()) == 1)
            .then(pl.lit(9_000_000, dtype=pl.UInt32))
            .otherwise(pl.col("count"))
            .alias("count")
        )
        np.testing.assert_array_equal(
            self._reference(counts, present, "presence"),
            self._reference(inflated, present, "presence"),
        )

    def test_abundance_does_not_ignore_it(self):
        """The same spike must move the abundance vectors, or the two metrics
        are not actually different."""
        counts, present = self._counts()
        inflated = counts.with_columns(
            pl.when(pl.int_range(pl.len()) == 1)
            .then(pl.lit(9_000_000, dtype=pl.UInt32))
            .otherwise(pl.col("count"))
            .alias("count")
        )
        self.assertFalse(
            np.array_equal(
                self._reference(counts, present, "abundance"),
                self._reference(inflated, present, "abundance"),
            )
        )

    def test_betasim_is_the_default(self):
        """Not presence/absence Bray-Curtis, which is Sorensen, and which Kreft
        & Jetz (2010) argue against for regionalisation because it is "strongly
        affected by differences in species richness" -- here largely a record of
        sampling effort rather than of biota."""
        from src import defaults
        from src.matrices.geocode_distance import betasim_condensed

        self.assertEqual(defaults.COMPOSITION_METRIC, "betasim")
        counts, present = self._counts()
        built = GeocodeDistanceMatrix.build(
            counts.lazy(), present.lazy(), metric="betasim"
        )
        presence_matrix = (
            build_unscaled_X(counts.lazy(), present.lazy()).to_numpy() > 0
        ).astype(float)
        reference = built.abundance_condensed()
        # Typed Optional because the abundance path can skip it; the betasim
        # path always sets it, which is the thing being asserted.
        assert reference is not None
        np.testing.assert_allclose(reference, betasim_condensed(presence_matrix))

    def test_presence_still_gives_sorensen(self):
        counts, present = self._counts()
        np.testing.assert_array_equal(
            self._reference(counts, present, "presence"),
            GeocodeDistanceMatrix.build(
                counts.lazy(), present.lazy(), metric="presence"
            ).abundance_condensed(),
        )


class TestThePivotRunsOnce(unittest.TestCase):
    """One pivot per build, on every metric.

    The pivot is one of the widest stages in the pipeline -- on the published
    extent it takes its own full pass at 10-15 GB -- and `presence` and
    `abundance` each ran it twice: once for the feature matrix and once more to
    build the reference matrix from an identical pivot. `betasim` returns
    before that point, which is why the default path never showed it and the
    logs did not either: only the scaled path was wrapped in `log_action`.
    """

    def _counts(self):
        counts = pl.DataFrame(
            {
                "geocode": ["a"] * 3 + ["b"] * 3 + ["c"] * 3 + ["d"] * 3,
                "taxonId": [1, 2, 3] * 4,
                "count": [5, 1, 0, 1, 7, 2, 0, 2, 9, 3, 3, 3],
            }
        )
        return counts, pl.DataFrame({"geocode": ["a", "b", "c", "d"]})

    def _build_counting_pivots(self, metric):
        counts, present = self._counts()
        real = geocode_distance.pivot_taxon_counts
        calls = []

        def counting(lf):
            calls.append(1)
            return real(lf)

        geocode_distance.pivot_taxon_counts = counting
        try:
            built = GeocodeDistanceMatrix.build(
                counts.lazy(),
                present.lazy(),
                random_state=0,
                metric=metric,
                reduction="pcoa",
            )
        finally:
            geocode_distance.pivot_taxon_counts = real
        return len(calls), built

    def test_every_metric_pivots_exactly_once(self):
        for metric in ("betasim", "presence", "abundance"):
            with self.subTest(metric=metric):
                pivots, _ = self._build_counting_pivots(metric)
                self.assertEqual(pivots, 1, f"{metric} pivoted {pivots} times")

    def test_reusing_the_pivot_does_not_change_the_distances(self):
        """The reference matrix must be what a second pivot would have given."""
        counts, present = self._counts()
        for metric in ("presence", "abundance"):
            with self.subTest(metric=metric):
                built = GeocodeDistanceMatrix.build(
                    counts.lazy(),
                    present.lazy(),
                    random_state=0,
                    metric=metric,
                    reduction="pcoa",
                )
                expected = build_unscaled_X(counts.lazy(), present.lazy()).to_numpy()
                if metric == "presence":
                    expected = (expected > 0).astype(float)
                reference = built.abundance_condensed()
                assert reference is not None
                np.testing.assert_allclose(
                    reference, pdist(np.ascontiguousarray(expected), metric="braycurtis")
                )
