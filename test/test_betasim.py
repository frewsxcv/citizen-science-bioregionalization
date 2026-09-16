"""Tests for betasim, the richness-independent turnover dissimilarity."""

import unittest

import numpy as np
from scipy.spatial.distance import pdist, squareform

from src.matrices.geocode_distance import betasim_condensed


class TestBetasim(unittest.TestCase):
    def _square(self, presence):
        return squareform(betasim_condensed(np.asarray(presence, dtype=float)))

    def test_a_nested_assemblage_is_not_a_different_one(self):
        """The property the whole change rests on. One hexagon's taxa being a
        subset of another's is what uneven sampling effort produces, and betasim
        scores it 0 where Sorensen scores it by the size of the gap."""
        rich = [1] * 100
        poor = [1] * 10 + [0] * 90
        square = self._square([rich, poor])
        self.assertAlmostEqual(square[0, 1], 0.0)

        sorensen = pdist(np.array([rich, poor], dtype=float), metric="braycurtis")[0]
        self.assertGreater(sorensen, 0.8)

    def test_disjoint_assemblages_are_maximally_dissimilar(self):
        square = self._square([[1, 1, 0, 0], [0, 0, 1, 1]])
        self.assertAlmostEqual(square[0, 1], 1.0)

    def test_identical_assemblages_are_identical(self):
        square = self._square([[1, 0, 1], [1, 0, 1]])
        self.assertAlmostEqual(square[0, 1], 0.0)

    def test_half_the_smaller_list_is_shared(self):
        square = self._square([[1, 1, 1, 1], [1, 1, 0, 0] and [1, 0, 0, 1]])
        # a=2 shared, b=2, c=0 -> min(b,c)=0 -> nested -> 0
        self.assertAlmostEqual(square[0, 1], 0.0)

    def test_a_genuine_turnover_scores_between(self):
        # a=1 shared, b=1, c=1 -> 1 - 1/(1+1) = 0.5
        square = self._square([[1, 1, 0], [1, 0, 1]])
        self.assertAlmostEqual(square[0, 1], 0.5)

    def test_it_is_a_valid_dissimilarity(self):
        rng = np.random.default_rng(0)
        presence = (rng.random((30, 200)) < 0.2).astype(float)
        square = self._square(presence)
        self.assertTrue(np.allclose(square, square.T))
        self.assertTrue(np.allclose(np.diag(square), 0))
        self.assertTrue(np.all((square >= 0) & (square <= 1)))

    def test_empty_assemblages_do_not_produce_nan(self):
        """0/0 is reachable: two hexagons sharing nothing and holding nothing."""
        square = self._square([[0, 0, 0], [0, 0, 0], [1, 1, 0]])
        self.assertTrue(np.all(np.isfinite(square)))
        self.assertAlmostEqual(square[0, 1], 0.0)

    def test_it_is_less_richness_dependent_than_sorensen(self):
        """Pairs drawn from one species pool but sampled to very different
        depths. betasim should call them similar; Sorensen should not."""
        rng = np.random.default_rng(1)
        pool = np.arange(500)
        rows = []
        for depth in (500, 400, 50, 25):
            row = np.zeros(500)
            row[rng.choice(pool, depth, replace=False)] = 1
            rows.append(row)
        presence = np.array(rows)
        bs = betasim_condensed(presence)
        sor = pdist(presence, metric="braycurtis")
        self.assertLess(bs.mean(), sor.mean())


if __name__ == "__main__":
    unittest.main()
