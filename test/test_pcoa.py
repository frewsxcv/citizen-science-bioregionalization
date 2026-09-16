"""Tests for the PCoA reduction that replaces UMAP."""

import unittest

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

from src.matrices.geocode_distance import reduce_dimensions_pcoa


def _community_matrix(seed: int = 0) -> np.ndarray:
    """A synthetic site-by-taxon matrix with three latent groups."""
    rng = np.random.default_rng(seed)
    x = np.zeros((90, 300))
    for i in range(90):
        group = i // 30
        x[i, group * 100 : (group + 1) * 100] = rng.poisson(5, 100)
        x[i] += rng.poisson(0.3, 300)
    return x


class TestPcoa(unittest.TestCase):
    def setUp(self):
        self.distances = pdist(_community_matrix(), metric="braycurtis")

    def test_it_is_reproducible(self):
        """The reason the reduction exists. UMAP's layout is a stochastic
        optimisation that a seed only pins within one machine: on the published
        run, two machines produced identical inputs and different embeddings,
        moving the reported R2 between 0.4591 and 0.5641."""
        a = reduce_dimensions_pcoa(self.distances, 16)
        b = reduce_dimensions_pcoa(self.distances, 16)
        np.testing.assert_array_equal(a, b)

    def test_it_preserves_the_dissimilarities(self):
        """What Ward is owed: Euclidean coordinates whose distances stand in for
        the Bray-Curtis ones they came from."""
        coords = reduce_dimensions_pcoa(self.distances, 32)
        embedded = pdist(coords, metric="euclidean")
        self.assertGreater(spearmanr(self.distances, embedded).statistic, 0.95)

    def test_more_axes_preserve_more(self):
        previous = 0.0
        for n_components in (2, 8, 32):
            coords = reduce_dimensions_pcoa(self.distances, n_components)
            fidelity = spearmanr(
                self.distances, pdist(coords, metric="euclidean")
            ).statistic
            self.assertGreater(fidelity, previous)
            previous = fidelity

    def test_it_drops_negative_eigenvalues(self):
        """Bray-Curtis is not Euclidean, so a double-centred Gram matrix always
        has some. Their axes have no real coordinates, so n_components is a
        ceiling rather than a promise."""
        coords = reduce_dimensions_pcoa(self.distances, 89)
        self.assertLessEqual(coords.shape[1], 89)
        self.assertTrue(np.all(np.isfinite(coords)))

    def test_axis_signs_are_pinned(self):
        """An eigenvector's sign is arbitrary and LAPACK need not pick the same
        one on every platform. Distances would not notice, but the digests this
        pipeline logs to localise a disagreement would."""
        coords = reduce_dimensions_pcoa(self.distances, 8)
        for axis in range(coords.shape[1]):
            column = coords[:, axis]
            with self.subTest(axis=axis):
                self.assertGreater(column[np.argmax(np.abs(column))], 0)

    def test_it_recovers_known_groups(self):
        from scipy.cluster.hierarchy import fcluster, linkage

        coords = reduce_dimensions_pcoa(self.distances, 16)
        labels = fcluster(linkage(pdist(coords), "ward"), 3, "maxclust")
        truth = np.repeat([0, 1, 2], 30)
        # Same partition up to relabelling: every true group falls in one cluster.
        for group in range(3):
            with self.subTest(group=group):
                self.assertEqual(len(set(labels[truth == group])), 1)


if __name__ == "__main__":
    unittest.main()
