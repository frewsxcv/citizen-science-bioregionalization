import unittest

import numpy as np
import polars as pl
from scipy.spatial.distance import pdist

from src.matrices.geocode_distance import GeocodeDistanceMatrix


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
