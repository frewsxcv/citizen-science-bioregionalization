"""Tests for the PERMANOVA wrapper, focusing on the R² effect size.

The p-value alone is misleading at the scale this pipeline runs at: with
thousands of geocodes almost any partition clears p < 0.001. R² reports how much
of the total dispersion the grouping actually explains.
"""

import unittest

import numpy as np
import polars as pl
from scipy.spatial.distance import pdist

import bioregion_rs

# Two tight clusters far apart.
POINTS = np.array(
    [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [10.0, 10.0], [10.1, 10.0], [10.0, 10.1]]
)
GEOCODES = list(range(len(POINTS)))


def _clusters(labels: list[int]) -> pl.DataFrame:
    return pl.DataFrame(
        {"geocode": GEOCODES, "cluster": labels},
        schema={"geocode": pl.UInt64, "cluster": pl.UInt32},
    )


def _run(labels: list[int]) -> dict:
    return bioregion_rs.build_permanova_results(
        pdist(POINTS).tolist(), GEOCODES, _clusters(labels), 99, 0
    ).to_dicts()[0]


class TestPermanovaRSquared(unittest.TestCase):
    def test_grouping_that_matches_the_structure_explains_nearly_everything(self):
        result = _run([0, 0, 0, 1, 1, 1])

        self.assertGreater(result["r_squared"], 0.99)

    def test_grouping_that_ignores_the_structure_explains_little(self):
        result = _run([0, 1, 0, 1, 0, 1])

        self.assertLess(result["r_squared"], 0.2)

    def test_r_squared_separates_cases_the_p_value_does_not(self):
        # Both groupings sit at an unremarkable p-value here, because six points
        # admit too few distinct permutations to reach significance. R² still
        # tells them apart, which is the reason for reporting it.
        good, bad = _run([0, 0, 0, 1, 1, 1]), _run([0, 1, 0, 1, 0, 1])

        self.assertGreater(good["r_squared"] - bad["r_squared"], 0.5)

    def test_seeded_runs_give_the_same_p_value(self):
        self.assertEqual(_run([0, 0, 0, 1, 1, 1])["p_value"], _run([0, 0, 0, 1, 1, 1])["p_value"])


if __name__ == "__main__":
    unittest.main()
