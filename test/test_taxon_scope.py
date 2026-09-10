import tempfile
import unittest
from pathlib import Path

import polars as pl

from src.darwin_core_utils import build_taxon_filter
from src.dataframes.darwin_core import build_darwin_core_lf
from src.taxon_scope import parse_scope
from src.types import TAXON_RANK_COLUMNS, TAXON_RANKS, Bbox, TaxonScope

SAMPLE_ARCHIVE = "test/sample-archive"
WORLD = Bbox.from_coordinates(-90.0, 90.0, -180.0, 180.0)


class TestParseScope(unittest.TestCase):
    def test_parses_rank_and_name(self):
        self.assertEqual(
            parse_scope("order:Coleoptera"), TaxonScope(rank="order", name="Coleoptera")
        )

    def test_empty_means_unscoped(self):
        for raw in ["", "   ", None]:
            self.assertIsNone(parse_scope(raw))

    def test_rank_is_case_insensitive(self):
        scope = parse_scope("ORDER:Coleoptera")
        assert scope is not None
        self.assertEqual(scope.rank, "order")

    def test_taxon_name_case_is_preserved(self):
        """Rank is normalised, the name is not: GBIF names are capitalised and
        the filter compares them literally."""
        scope = parse_scope("class:Aves")
        assert scope is not None
        self.assertEqual(scope.name, "Aves")

    def test_surrounding_whitespace_tolerated(self):
        self.assertEqual(parse_scope("  class : Aves  "), parse_scope("class:Aves"))

    def test_missing_separator_rejected(self):
        with self.assertRaisesRegex(ValueError, "Malformed scope"):
            parse_scope("Coleoptera")

    def test_unknown_rank_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported rank"):
            parse_scope("tribe:Bombini")

    def test_missing_name_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing a taxon name"):
            parse_scope("class:")

    def test_backbone_key_rejected_with_guidance(self):
        """The old rank:key form must fail loudly rather than search for a
        taxon literally named '1470' and report an empty run."""
        with self.assertRaises(ValueError) as ctx:
            parse_scope("order:1470")
        message = str(ctx.exception)
        self.assertIn("backbone key", message)
        self.assertIn("order:Aves", message)

    def test_scope_str_is_the_input_form(self):
        self.assertEqual(str(parse_scope("class:Aves")), "class:Aves")

    def test_every_rank_has_a_column(self):
        for rank in TAXON_RANKS:
            self.assertIn(rank, TAXON_RANK_COLUMNS)


class TestBuildTaxonFilter(unittest.TestCase):
    def test_filters_on_rank_name_column(self):
        scope = TaxonScope(rank="order", name="Coleoptera")
        df = pl.DataFrame(
            {
                "order": ["Coleoptera", "Diptera", "Coleoptera", None],
                "n": [1, 2, 3, 4],
            }
        )
        self.assertEqual(df.filter(build_taxon_filter(scope))["n"].to_list(), [1, 3])

    def test_null_rank_name_is_excluded(self):
        """A record identified only to a coarser rank has a null name there and
        is not known to belong to the scope, so it must be dropped."""
        scope = TaxonScope(rank="order", name="Coleoptera")
        df = pl.DataFrame({"order": pl.Series([None, None], dtype=pl.String)})
        self.assertEqual(df.filter(build_taxon_filter(scope)).height, 0)

    def test_matching_is_case_sensitive(self):
        """GBIF capitalises names above species; a lowercase scope matches
        nothing rather than silently widening."""
        df = pl.DataFrame({"class": ["Aves", "aves"], "n": [1, 2]})
        exact = df.filter(build_taxon_filter(TaxonScope(rank="class", name="Aves")))
        self.assertEqual(exact["n"].to_list(), [1])


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

    def test_scope_applied_before_limit(self):
        """Regression test for an ordering bug: limiting first and scoping
        second yields near-zero rows for any selective scope."""
        total_aves = self._load("class:Aves").height
        self.assertGreater(total_aves, 1000, "fixture too small to exercise ordering")
        self.assertEqual(self._load("class:Aves", limit=1000).height, 1000)

    def test_scoping_does_not_change_schema(self):
        """Downstream stages must not need to know whether a run was scoped."""
        self.assertEqual(self._load(None).columns, self._load("class:Aves").columns)

    def test_unknown_name_yields_no_rows(self):
        """The data is the authority. A name that is not in it matches nothing,
        rather than being rejected against a registry that has to be kept in
        step with the GBIF backbone."""
        self.assertEqual(self._load("class:Nonexistentia").height, 0)

    def test_missing_rank_column_raises(self):
        """Silently skipping the filter is the exact failure mode being fixed."""
        df = pl.DataFrame(
            {
                "decimalLatitude": [1.0],
                "decimalLongitude": [1.0],
                "scientificName": ["x"],
                "taxonKey": pl.Series(["3DTGL"], dtype=pl.String),
                "individualCount": pl.Series([1], dtype=pl.Int32),
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "occurrence.parquet"
            df.write_parquet(path)
            with self.assertRaisesRegex(ValueError, "has no 'order' column"):
                build_darwin_core_lf(
                    path, bounding_box=WORLD, scope=parse_scope("order:Coleoptera")
                )


if __name__ == "__main__":
    unittest.main()
