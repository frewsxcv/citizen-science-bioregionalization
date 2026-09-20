"""Tests for emitting several cuts of the merge tree rather than one."""

import unittest

from src.hierarchy import (
    default_ladder,
    resolve_cluster_levels,
    resolve_default_level,
    resolve_levels,
)


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


class TestDefaultLadder(unittest.TestCase):
    """The nesting emitted when no levels are asked for.

    Before this the default was the selector's k alone, so the published
    aggregations.json carried exactly one level and the level selector had
    nothing to select from.
    """

    def test_doubles_rather_than_stepping_by_one(self) -> None:
        """Consecutive cuts differ by one split, which is not a change of grain."""
        self.assertEqual(default_ladder(2, 15), [2, 4, 8])

    def test_stays_inside_the_range_the_tree_was_cut_at(self) -> None:
        self.assertEqual(default_ladder(2, 10), [2, 4, 8])
        self.assertEqual(default_ladder(3, 20), [3, 6, 12])

    def test_never_returns_nothing(self) -> None:
        """A range too narrow to double in still has to yield a level."""
        self.assertEqual(default_ladder(2, 3), [2])
        self.assertEqual(default_ladder(5, 5), [5])

    def test_combines_with_the_display_level(self) -> None:
        levels = resolve_levels(
            default_ladder(2, 15), optimal=2, min_k=2, max_k=15, display=4
        )
        self.assertEqual(levels, [2, 4, 8])
        self.assertEqual(resolve_default_level(4, 2, levels), 4)


class TestResolveClusterLevels(unittest.TestCase):
    """The single entry point the notebook calls.

    Exists so the decision is made once. It previously took two calls whose
    results travelled downstream as loose ints, and the published cut picked up
    a different name at each module boundary.
    """

    def test_publishes_the_preferred_level_not_the_selector_s(self) -> None:
        levels = resolve_cluster_levels(
            None, selector_k=2, min_k=2, max_k=15, preferred_display=4
        )
        self.assertEqual(levels.published, 4)
        self.assertEqual(levels.selector, 2)
        self.assertFalse(levels.selector_agrees)

    def test_the_published_cut_is_always_emitted(self) -> None:
        """A consumer cannot open on a level that is not in the document."""
        for preferred in (2, 3, 4, 8, 15):
            levels = resolve_cluster_levels(
                None, selector_k=2, min_k=2, max_k=15, preferred_display=preferred
            )
            self.assertIn(levels.published, levels.emitted)

    def test_the_selector_s_cut_is_always_emitted(self) -> None:
        levels = resolve_cluster_levels(
            [8], selector_k=3, min_k=2, max_k=15, preferred_display=4
        )
        self.assertIn(3, levels.emitted)
        self.assertIn(4, levels.emitted)
        self.assertIn(8, levels.emitted)

    def test_falls_back_to_the_selector_when_the_preferred_is_out_of_range(self) -> None:
        levels = resolve_cluster_levels(
            None, selector_k=2, min_k=2, max_k=3, preferred_display=4
        )
        self.assertEqual(levels.published, 2)
        self.assertIn(2, levels.emitted)

    def test_no_requested_levels_uses_the_doubling_ladder(self) -> None:
        levels = resolve_cluster_levels(
            None, selector_k=2, min_k=2, max_k=15, preferred_display=4
        )
        self.assertEqual(levels.emitted, (2, 4, 8))

    def test_an_empty_request_means_nothing_asked_for(self) -> None:
        """`--hierarchy-levels=` parses to [], which is not "emit nothing"."""
        self.assertEqual(
            resolve_cluster_levels(
                [], selector_k=2, min_k=2, max_k=15, preferred_display=4
            ).emitted,
            (2, 4, 8),
        )

    def test_records_the_range_and_whether_k_was_pinned(self) -> None:
        levels = resolve_cluster_levels(
            None, selector_k=6, min_k=2, max_k=15, preferred_display=4, pinned=True
        )
        self.assertEqual((levels.min_k, levels.max_k), (2, 15))
        self.assertTrue(levels.pinned)

    def test_emitted_is_sorted_and_hashable(self) -> None:
        """A tuple, so the decision cannot be mutated after it is made."""
        levels = resolve_cluster_levels(
            [8, 2], selector_k=3, min_k=2, max_k=15, preferred_display=4
        )
        self.assertEqual(levels.emitted, tuple(sorted(levels.emitted)))
        hash(levels)
