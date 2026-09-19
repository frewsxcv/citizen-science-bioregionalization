"""Tests for emitting several cuts of the merge tree rather than one."""

import unittest

from src.hierarchy import resolve_default_level, resolve_levels


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


class TestDefaultLevel(unittest.TestCase):
    """Which cut a consumer opens on.

    Deliberately not the selector's. On the published run the selector took
    k=2, and both references the run computes put that cut last: the two clades
    agree at ARI -0.005 there, and agreement with EPA Level II is its lowest.
    """

    def test_emits_the_display_level_so_it_can_be_opened(self) -> None:
        """With no --hierarchy-levels the selector's k is otherwise the only one."""
        levels = resolve_levels(None, optimal=2, min_k=2, max_k=15, display=4)
        self.assertEqual(levels, [2, 4])

    def test_opens_on_the_display_level_rather_than_the_selector_s(self) -> None:
        levels = resolve_levels(None, optimal=2, min_k=2, max_k=15, display=4)
        self.assertEqual(resolve_default_level(4, 2, levels), 4)

    def test_falls_back_to_the_selector_when_the_level_is_out_of_range(self) -> None:
        """A default level must always name a level that was emitted."""
        levels = resolve_levels(None, optimal=2, min_k=2, max_k=3, display=4)
        self.assertNotIn(4, levels)
        self.assertEqual(resolve_default_level(4, 2, levels), 2)

    def test_display_level_does_not_displace_requested_levels(self) -> None:
        levels = resolve_levels([2, 8], optimal=3, min_k=2, max_k=15, display=4)
        self.assertEqual(levels, [2, 3, 4, 8])
