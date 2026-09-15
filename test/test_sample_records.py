"""Tests for the uniform record sample that replaces the scan-order cap."""

import unittest

import polars as pl

from src.dataframes.darwin_core import sample_records_lf


class TestSampleRecords(unittest.TestCase):
    """`limit` keeps the head of the scan; this keeps a uniform sample.

    The distinction is load-bearing at snapshot scale. A GBIF snapshot's file
    order is grouped by source dataset, so the first N rows of the published
    bounding box are whichever datasets sort earliest -- 54.4% of that bbox was
    being discarded that way.
    """

    def _frame(self, n: int) -> pl.LazyFrame:
        return pl.LazyFrame({"value": range(n)})

    def test_keeps_roughly_the_target(self):
        kept = sample_records_lf(self._frame(100_000), 10_000).collect().height
        # Binomial around the target, not equal to it.
        self.assertAlmostEqual(kept, 10_000, delta=500)

    def test_is_reproducible(self):
        """Determinism is load-bearing: two runs on one input must produce
        byte-identical output, and this decides which records a run reads."""
        runs = [sample_records_lf(self._frame(50_000), 5_000).collect() for _ in range(2)]
        self.assertTrue(runs[0].equals(runs[1]))

    def test_a_different_seed_draws_a_different_sample(self):
        a = sample_records_lf(self._frame(50_000), 5_000, seed=0).collect()
        b = sample_records_lf(self._frame(50_000), 5_000, seed=1).collect()
        self.assertFalse(a.equals(b))

    def test_a_frame_under_the_target_is_untouched(self):
        kept = sample_records_lf(self._frame(100), 1_000).collect()
        self.assertEqual(kept.height, 100)
        self.assertEqual(kept["value"].to_list(), list(range(100)))

    def test_the_sample_is_spread_across_the_frame(self):
        """The point of the change. A head-of-frame cap would put every kept
        row in the first decile; a uniform sample spreads them evenly, so no
        source dataset is over-represented by virtue of sort order."""
        n, target = 100_000, 10_000
        kept = sample_records_lf(self._frame(n), target).collect()["value"].to_numpy()
        deciles = [((kept >= i * n / 10) & (kept < (i + 1) * n / 10)).sum() for i in range(10)]
        for i, count in enumerate(deciles):
            with self.subTest(decile=i):
                self.assertAlmostEqual(count, target / 10, delta=200)

    def test_it_does_not_reorder_what_it_keeps(self):
        kept = sample_records_lf(self._frame(20_000), 2_000).collect()["value"].to_list()
        self.assertEqual(kept, sorted(kept))


if __name__ == "__main__":
    unittest.main()
