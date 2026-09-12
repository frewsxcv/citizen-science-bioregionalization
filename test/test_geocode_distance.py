import unittest

import numpy as np
import polars as pl
from scipy.spatial.distance import pdist

from src.matrices.geocode_distance import GeocodeDistanceMatrix


class TestDistancesAreBrayCurtisOnCounts(unittest.TestCase):
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

    def test_distances_match_bray_curtis_on_the_stored_features(self):
        counts, present = self._counts()
        matrix = GeocodeDistanceMatrix.build(counts.lazy(), present.lazy())
        np.testing.assert_allclose(
            matrix.condensed(),
            pdist(matrix.features(), metric="braycurtis"),
            rtol=1e-9,
            atol=1e-9,
        )

    def test_distances_stay_within_the_bray_curtis_bound(self):
        """Bray-Curtis is bounded on [0, 1] for non-negative input, and the
        counts are never scaled before this point, so nothing goes negative."""
        counts, present = self._counts()
        matrix = GeocodeDistanceMatrix.build(counts.lazy(), present.lazy())
        self.assertGreaterEqual(matrix.condensed().min(), 0.0)
        self.assertLessEqual(matrix.condensed().max(), 1.0)
        self.assertGreaterEqual(matrix.features().min(), 0.0)

    def test_direct_construction_still_works(self):
        """Test fixtures build the matrix directly; that must stay possible."""
        matrix = GeocodeDistanceMatrix(np.array([1.0]), np.zeros((2, 2)))
        self.assertEqual(matrix.condensed().shape, (1,))
