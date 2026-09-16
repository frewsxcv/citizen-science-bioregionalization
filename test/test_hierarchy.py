"""Tests for emitting several cuts of the merge tree rather than one."""

import unittest

from src.hierarchy import resolve_levels


class TestResolveLevels(unittest.TestCase):
    def test_the_selected_level_is_always_present(self):
        """Single-level consumers must still find the cut the selector chose,
        whatever else was asked for."""
        self.assertIn(3, resolve_levels([5, 8], optimal=3, min_k=2, max_k=15))

    def test_none_gives_just_the_selected_level(self):
        self.assertEqual(resolve_levels(None, optimal=4, min_k=2, max_k=15), [4])

    def test_levels_are_sorted_and_deduplicated(self):
        self.assertEqual(
            resolve_levels([8, 2, 8, 4], optimal=4, min_k=2, max_k=15), [2, 4, 8]
        )

    def test_levels_outside_the_tree_are_dropped(self):
        """The tree is only cut between min_k and max_k, so a level outside that
        range has nothing to extract."""
        self.assertEqual(
            resolve_levels([1, 4, 99], optimal=4, min_k=2, max_k=15), [4]
        )

    def test_the_selected_level_survives_the_range_check(self):
        self.assertEqual(resolve_levels([99], optimal=2, min_k=2, max_k=15), [2])


if __name__ == "__main__":
    unittest.main()
