import tempfile
import unittest
from pathlib import Path

import polars as pl

from src.darwin_core_utils import build_taxon_filter
from src.dataframes.darwin_core import build_darwin_core_lf
from src.taxon_scope import known_names, load_registry, parse_scope
from src.types import TAXON_RANK_COLUMNS, TAXON_RANKS, Bbox, TaxonScope

SAMPLE_ARCHIVE = "test/sample-archive"
WORLD = Bbox.from_coordinates(-90.0, 90.0, -180.0, 180.0)


class TestParseScope(unittest.TestCase):
    def test_parses_rank_and_name(self):
        scope = parse_scope("order:Coleoptera")
        self.assertEqual(scope, TaxonScope(rank="order", key=1470, label="Coleoptera"))

    def test_parses_bare_key(self):
        """A numeric name is taken as a backbone key, so taxa outside the
        curated registry are still reachable without a network lookup."""
        scope = parse_scope("order:1470")
        assert scope is not None
        self.assertEqual(scope.rank, "order")
        self.assertEqual(scope.key, 1470)

    def test_name_and_key_forms_agree(self):
        by_name = parse_scope("class:Aves")
        by_key = parse_scope("class:212")
        assert by_name is not None and by_key is not None
        self.assertEqual(by_name.key, by_key.key)

    def test_empty_means_unscoped(self):
        for raw in ["", "   ", None]:
            self.assertIsNone(parse_scope(raw))

    def test_rank_is_case_insensitive(self):
        scope = parse_scope("ORDER:Coleoptera")
        assert scope is not None
        self.assertEqual(scope.rank, "order")

    def test_surrounding_whitespace_tolerated(self):
        self.assertEqual(parse_scope("  order : Coleoptera  "), parse_scope("order:Coleoptera"))

    def test_missing_separator_rejected(self):
        with self.assertRaisesRegex(ValueError, "Malformed scope"):
            parse_scope("Aves")

    def test_unknown_rank_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported rank"):
            parse_scope("realm:Aves")

    def test_missing_name_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing a taxon name or key"):
            parse_scope("order:")

    def test_unknown_name_rejected_with_guidance(self):
        """An unresolvable name must fail loudly rather than silently running
        unscoped, and should say how to proceed."""
        with self.assertRaises(ValueError) as ctx:
            parse_scope("order:Notarealorder")
        message = str(ctx.exception)
        self.assertIn("Unknown order", message)
        self.assertIn("scripts/fetch_taxon_keys.py", message)

    def test_scope_str_includes_key(self):
        self.assertEqual(str(parse_scope("class:Aves")), "class:Aves(212)")


class TestRegistry(unittest.TestCase):
    def test_registry_ranks_are_supported(self):
        for rank in load_registry():
            self.assertIn(rank, TAXON_RANKS)

    def test_registry_keys_are_positive_ints(self):
        for rank, names in load_registry().items():
            for name, key in names.items():
                self.assertIsInstance(key, int, f"{rank}:{name}")
                self.assertGreater(key, 0, f"{rank}:{name}")

    def test_every_rank_has_a_key_column(self):
        for rank in TAXON_RANKS:
            self.assertIn(rank, TAXON_RANK_COLUMNS)

    def test_known_names_sorted(self):
        names = known_names("kingdom")
        self.assertEqual(names, sorted(names))
        self.assertIn("Animalia", names)

    def test_backbone_rank_quirks_are_respected(self):
        """GBIF's backbone disagrees with textbook taxonomy in places. Squamata
        is a backbone class, not an order, and Reptilia is absent entirely
        (paraphyletic). Guard against a well-meaning edit reintroducing them at
        the intuitive-but-wrong rank."""
        self.assertIn("Squamata", known_names("class"))
        self.assertNotIn("Squamata", known_names("order"))
        self.assertNotIn("Reptilia", known_names("class"))


class TestBuildTaxonFilter(unittest.TestCase):
    def test_filters_on_rank_key_column(self):
        scope = TaxonScope(rank="order", key=1470, label="Coleoptera")
        df = pl.DataFrame(
            {
                "orderKey": pl.Series([1470, 797, 1470, None], dtype=pl.UInt32),
                "n": [1, 2, 3, 4],
            }
        )
        self.assertEqual(df.filter(build_taxon_filter(scope))["n"].to_list(), [1, 3])

    def test_null_rank_key_is_excluded(self):
        """A record identified only to a coarser rank has a null key there and
        is not known to belong to the scope, so it must be dropped."""
        scope = TaxonScope(rank="order", key=1470, label="Coleoptera")
        df = pl.DataFrame({"orderKey": pl.Series([None, None], dtype=pl.UInt32)})
        self.assertEqual(df.filter(build_taxon_filter(scope)).height, 0)


class TestScopedLoad(unittest.TestCase):
    """End-to-end scoping against the checked-in Darwin Core archive."""

    def _load(self, scope_str, limit=None):
        return build_darwin_core_lf(
            SAMPLE_ARCHIVE,
            bounding_box=WORLD,
            limit=limit,
            scope=parse_scope(scope_str),
        ).collect()

    def test_scoping_narrows_results(self):
        unscoped = self._load(None).height
        animalia = self._load("kingdom:Animalia").height
        plantae = self._load("kingdom:Plantae").height

        self.assertGreater(animalia, 0)
        self.assertGreater(plantae, 0)
        self.assertLess(animalia, unscoped)
        self.assertLess(plantae, unscoped)
        # Kingdoms are disjoint, so they cannot jointly exceed the whole.
        self.assertLessEqual(animalia + plantae, unscoped)

    def test_nested_scopes_are_consistent(self):
        """Aves is within Animalia, so it cannot be the larger set."""
        self.assertLessEqual(
            self._load("class:Aves").height, self._load("kingdom:Animalia").height
        )

    def test_name_and_key_forms_load_identically(self):
        self.assertEqual(
            self._load("order:Coleoptera").height, self._load("order:1470").height
        )

    def test_scope_applied_before_limit(self):
        """Regression test for the ordering bug this replaces: limiting first
        and scoping second yields near-zero rows for any selective scope."""
        total_aves = self._load("class:Aves").height
        self.assertGreater(total_aves, 1000, "fixture too small to exercise ordering")
        self.assertEqual(self._load("class:Aves", limit=1000).height, 1000)

    def test_scoping_does_not_change_schema(self):
        """Downstream stages must not need to know whether a run was scoped."""
        self.assertEqual(
            self._load(None).columns, self._load("class:Aves").columns
        )

    def test_missing_rank_column_raises(self):
        """Silently skipping the filter is the exact failure mode being fixed."""
        df = pl.DataFrame(
            {
                "decimalLatitude": [1.0],
                "decimalLongitude": [1.0],
                "scientificName": ["x"],
                "taxonKey": pl.Series([1], dtype=pl.UInt32),
                "individualCount": pl.Series([1], dtype=pl.Int32),
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "occurrence.parquet"
            df.write_parquet(path)
            with self.assertRaisesRegex(ValueError, "has no 'orderKey' column"):
                build_darwin_core_lf(
                    path, bounding_box=WORLD, scope=parse_scope("order:Coleoptera")
                )


if __name__ == "__main__":
    unittest.main()
