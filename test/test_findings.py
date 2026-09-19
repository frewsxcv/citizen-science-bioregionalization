"""
Unit tests for the generated findings.

The page these produce used to be written by hand, with the numbers transcribed
from runs, so the thing worth guarding is that a section carries this run's
numbers or says it computed none -- never a number from somewhere else. Most of
these assert the degraded paths for that reason.
"""

import json
import unittest

import polars as pl
import polars_h3

from src.clade_congruence import Congruence, clade_taxon_ids, congruence_by_k
from src.epa_reference import reference_region_lf, score_against_reference
from src.findings import (
    build_findings_data,
    choose_span_cut,
    clade_shares,
    reference_agreement_by_k,
)
from src.findings_page import (
    CladeShare,
    FindingsData,
    RunContext,
    findings_summary_json,
    render_findings_page,
)


def a_context(**overrides: object) -> RunContext:
    base: dict[str, object] = {
        "source": "test/sample-archive",
        "bbox": "25-47N, 87-66W",
        "geocode_precision": 4,
        "hexagons": 2098,
        "taxa": 208296,
        "taxa_analysed": 10000,
        "records": 1_000_000,
        "chosen_k": 2,
        "composition_metric": "betasim",
        "seed": 0,
    }
    base.update(overrides)
    return RunContext(**base)  # type: ignore[arg-type]


def geocodes_for(points: list[tuple[float, float]], resolution: int = 4) -> pl.DataFrame:
    return (
        pl.DataFrame({"lat": [p[0] for p in points], "lng": [p[1] for p in points]})
        .with_columns(
            geocode=polars_h3.latlng_to_cell(
                "lat", "lng", resolution, return_dtype=pl.UInt64
            )
        )
        .select("geocode")
    )


class TestCladeShares(unittest.TestCase):
    def _counts(self) -> pl.LazyFrame:
        return pl.DataFrame(
            {
                "geocode": [1, 1, 2, 2],
                "taxonId": [10, 20, 30, 40],
                "count": [900, 10, 80, 10],
            }
        ).lazy()

    def _clades(self) -> pl.LazyFrame:
        return pl.DataFrame(
            {
                "taxonId": [10, 20, 30, 40],
                "kingdom": ["Animalia", "Plantae", "Plantae", "Fungi"],
                "class": ["Aves", "Magnoliopsida", "Pinopsida", None],
            }
        ).lazy()

    def test_reports_the_inversion_between_records_and_taxa(self) -> None:
        """One clade can dominate the records and almost none of the species."""
        shares = {s.name: s for s in clade_shares(self._counts(), self._clades())}

        self.assertAlmostEqual(shares["Aves"].record_share, 900 / 1000)
        self.assertAlmostEqual(shares["Aves"].taxon_share, 1 / 4)
        self.assertAlmostEqual(shares["Plantae"].record_share, 90 / 1000)
        self.assertAlmostEqual(shares["Plantae"].taxon_share, 2 / 4)

    def test_returns_nothing_without_clade_columns(self) -> None:
        """A source with no rank columns yields no clade section, not a zero."""
        self.assertEqual(clade_shares(self._counts(), None), [])

    def test_omits_a_clade_with_no_records(self) -> None:
        clades = self._clades().with_columns(
            pl.when(pl.col("class") == "Aves")
            .then(pl.lit("Mammalia"))
            .otherwise(pl.col("class"))
            .alias("class")
        )
        names = {s.name for s in clade_shares(self._counts(), clades)}
        self.assertNotIn("Aves", names)

    def test_selects_by_the_rank_the_clade_is_named_at(self) -> None:
        ids = clade_taxon_ids(self._clades(), "kingdom", "Plantae").collect()
        self.assertEqual(sorted(ids["taxonId"].to_list()), [20, 30])


class TestReferenceScoring(unittest.TestCase):
    # Carolinas and coastal Georgia against New England: a real north/south
    # contrast inside the checked-in extent.
    SOUTH = [(35.8, -79.0), (34.5, -80.0), (32.0, -81.5), (31.5, -82.0)]
    NORTH = [(44.0, -72.0), (43.0, -71.0), (44.5, -70.0), (45.0, -69.5)]

    def test_assigns_east_coast_hexagons_a_level_two_region(self) -> None:
        df = geocodes_for(self.SOUTH + self.NORTH)
        assigned = reference_region_lf(df.lazy()).collect()

        self.assertEqual(assigned.height, 8)
        self.assertEqual(assigned.drop_nulls("reference").height, 8)

    def test_scores_a_partition_that_matches_the_framework_above_one_that_does_not(
        self,
    ) -> None:
        df = geocodes_for(self.SOUTH + self.NORTH)
        reference = reference_region_lf(df.lazy()).collect().lazy()

        geographic = df.with_columns(cluster=pl.Series([0, 0, 0, 0, 1, 1, 1, 1]))
        interleaved = df.with_columns(cluster=pl.Series([0, 1, 0, 1, 0, 1, 0, 1]))

        good = score_against_reference(geographic, reference)
        bad = score_against_reference(interleaved, reference)
        assert good is not None and bad is not None
        self.assertGreater(good.adjusted_rand, bad.adjusted_rand)

    def test_returns_nothing_outside_the_reference_extent(self) -> None:
        """Monaco is not on the US East Coast, and saying nothing beats a zero."""
        df = geocodes_for([(43.73, 7.42), (43.75, 7.43)])
        reference = reference_region_lf(df.lazy()).collect().lazy()
        clusters = df.with_columns(cluster=pl.Series([0, 1]))

        self.assertIsNone(score_against_reference(clusters, reference))

    def test_skips_cuts_the_run_did_not_fit(self) -> None:
        df = geocodes_for(self.SOUTH + self.NORTH)
        reference = reference_region_lf(df.lazy()).collect().lazy()
        multi_k = df.with_columns(
            cluster=pl.Series([0, 0, 0, 0, 1, 1, 1, 1]), num_clusters=pl.lit(2)
        )

        scored = reference_agreement_by_k(multi_k, reference, [2, 4, 8])
        self.assertEqual([k for k, _ in scored], [2])


class TestCongruence(unittest.TestCase):
    def _partition(self, clusters: dict[int, list[int]]) -> object:
        from src.clade_congruence import CladePartition

        rows = [
            {"geocode": g, "cluster": c, "num_clusters": k}
            for k, assignment in clusters.items()
            for g, c in enumerate(assignment)
        ]
        return CladePartition(
            name="x", multi_k_df=pl.DataFrame(rows), geocodes=4, taxa=10
        )

    def test_identical_partitions_agree_perfectly(self) -> None:
        a = self._partition({2: [0, 0, 1, 1]})
        b = self._partition({2: [1, 1, 0, 0]})  # same split, relabelled
        result = congruence_by_k(a, b)  # type: ignore[arg-type]

        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0].adjusted_rand, 1.0)

    def test_compares_only_cuts_both_clades_fit(self) -> None:
        a = self._partition({2: [0, 0, 1, 1], 3: [0, 1, 2, 2]})
        b = self._partition({2: [0, 0, 1, 1]})
        self.assertEqual([c.num_clusters for c in congruence_by_k(a, b)], [2])  # type: ignore[arg-type]


class TestRendering(unittest.TestCase):
    def test_says_so_when_nothing_was_computed(self) -> None:
        """The failure this replaces is a stale number, not a missing one."""
        page = render_findings_page(FindingsData(context=a_context()))

        self.assertIn("Not computed in this run", page)
        self.assertNotIn("0.000", page)

    def test_draws_every_section_it_has_data_for(self) -> None:
        data = FindingsData(
            context=a_context(),
            clade_shares=[CladeShare("Aves", "aves", 0.914, 0.021, 900, 21)],
            congruence=[Congruence(2, 0.318, 1800), Congruence(8, 0.460, 1750)],
            congruence_pair=("Aves", "Plantae"),
            latitude_spans={"Aves": [(0, 31.4, 47.0, 1200), (1, 25.2, 31.4, 600)]},
        )
        page = render_findings_page(data)

        self.assertIn("0.460", page)
        self.assertIn("91.4% of records", page)
        self.assertIn("<svg", page)
        # Every chart carrying numbers gets a table behind a disclosure.
        self.assertGreaterEqual(page.count("Show as table"), 2)

    def test_describes_each_chart_for_a_screen_reader(self) -> None:
        data = FindingsData(
            context=a_context(),
            congruence=[Congruence(2, 0.318, 1800)],
            congruence_pair=("Aves", "Plantae"),
        )
        page = render_findings_page(data)
        self.assertIn('role="img"', page)
        self.assertIn("aria-label", page)

    def test_escapes_values_that_come_from_the_command_line(self) -> None:
        data = FindingsData(context=a_context(source='<script>x</script>'))
        page = render_findings_page(data)
        self.assertNotIn("<script>", page)

    def test_reports_both_taxa_counts(self) -> None:
        """The shares are fractions of the analysed set, not of the taxonomy.

        Showing only the larger number invites dividing by it: 671 Aves taxa is
        6.7% of 10,000 analysed and 0.3% of 208,296 in the taxonomy.
        """
        data = FindingsData(
            context=a_context(),
            clade_shares=[CladeShare("Aves", "aves", 0.965, 0.0671, 900, 671)],
        )
        page = render_findings_page(data)

        self.assertIn("10,000 taxa", page)
        self.assertIn("208,296", page)

    def test_names_the_cut_it_opens_on_when_it_differs(self) -> None:
        opens_elsewhere = render_findings_page(
            FindingsData(context=a_context(chosen_k=2, display_k=4))
        )
        self.assertIn("opens on <strong>4</strong>", opens_elsewhere)

        same = render_findings_page(
            FindingsData(context=a_context(chosen_k=4, display_k=4))
        )
        self.assertNotIn("opens on", same)

    def test_summary_json_round_trips(self) -> None:
        data = FindingsData(
            context=a_context(),
            congruence=[Congruence(2, 0.318, 1800)],
            clade_shares=[CladeShare("Aves", "aves", 0.9, 0.02, 900, 21)],
        )
        parsed = json.loads(findings_summary_json(data))

        self.assertEqual(parsed["congruence"][0]["adjusted_rand"], 0.318)
        self.assertEqual(parsed["context"]["chosen_k"], 2)


if __name__ == "__main__":
    unittest.main()


class TestPublishedCut(unittest.TestCase):
    """Which cut the page's figures describe.

    This was wrong once already: the latitude-span chart was drawn at the
    selector's k while every other figure and every artifact described the
    published level, so the deployed page showed the degenerate Aves 1182/4
    split from k=2 beside numbers for k=4.
    """

    def test_published_cut_is_the_display_level_when_there_is_one(self) -> None:
        self.assertEqual(a_context(chosen_k=2, display_k=4).published_k, 4)

    def test_published_cut_falls_back_to_the_selector(self) -> None:
        """A run that never set a display level publishes the selector's."""
        self.assertEqual(a_context(chosen_k=3, display_k=None).published_k, 3)

    def test_scores_the_cut_it_publishes(self) -> None:
        """A published level outside the fixed spread must still be scored."""
        df = geocodes_for(
            TestReferenceScoring.SOUTH + TestReferenceScoring.NORTH
        )
        reference = reference_region_lf(df.lazy()).collect().lazy()
        multi_k = pl.concat(
            [
                df.with_columns(
                    cluster=pl.Series([0, 0, 0, 0, 1, 1, 1, 1]),
                    num_clusters=pl.lit(k, dtype=pl.Int64),
                )
                for k in (2, 5)
            ]
        )
        data = build_findings_data(
            a_context(chosen_k=2, display_k=5),
            pl.DataFrame({"geocode": [], "taxonId": [], "count": []}).lazy(),
            df.lazy(),
            multi_k,
            None,
            min_k=2,
            max_k=6,
            seed=0,
            metric="betasim",
            reduction="pcoa",
        )

        scored = [k for k, _ in data.reference_by_k]
        self.assertIn(5, scored, "the published cut was never scored")
        self.assertIn(2, scored, "the selector's cut is kept for comparison")

    def test_spans_are_drawn_at_the_published_cut(self) -> None:
        self.assertEqual(choose_span_cut(4, {2, 3, 4, 5, 8}), 4)

    def test_spans_fall_back_to_the_nearest_cut_the_clade_was_fit_at(self) -> None:
        """A sparse clade may not have been fit at the published level."""
        self.assertEqual(choose_span_cut(8, {2, 3, 4}), 4)
        self.assertEqual(choose_span_cut(4, {2, 6}), 2)

    def test_validation_prose_leads_with_the_published_cut(self) -> None:
        from src.epa_reference import ReferenceAgreement

        data = FindingsData(
            context=a_context(chosen_k=2, display_k=4),
            reference_by_k=[
                (k, ReferenceAgreement(a, v, 1071, 116))
                for k, a, v in [(2, 0.152, 0.315), (4, 0.315, 0.452)]
            ],
        )
        page = render_findings_page(data)

        self.assertIn("which is the cut this run publishes", page)
        self.assertIn("The selector's 2 scores 0.152", page)

    def test_validation_prose_when_the_peak_is_not_published(self) -> None:
        from src.epa_reference import ReferenceAgreement

        data = FindingsData(
            context=a_context(chosen_k=2, display_k=4),
            reference_by_k=[
                (k, ReferenceAgreement(a, v, 1071, 116))
                for k, a, v in [(2, 0.152, 0.315), (4, 0.290, 0.452), (8, 0.315, 0.473)]
            ],
        )
        page = render_findings_page(data)

        self.assertIn("this run publishes 4 (ARI 0.290)", page)

    def test_caption_names_the_cut(self) -> None:
        page = render_findings_page(
            FindingsData(
                context=a_context(chosen_k=2, display_k=4),
                latitude_spans={"Aves": [(0, 31.4, 47.0, 1200)]},
            )
        )
        self.assertIn("at 4 regions, the cut this run publishes", page)


