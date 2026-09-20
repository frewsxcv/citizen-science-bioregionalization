"""Tests for the resolved run configuration.

The point of RunConfig is that a run's settings exist in one place, so the
things worth asserting are that nothing is lost on the way in and that the
notebook's inputs table reports all of it.
"""

import dataclasses
import unittest

from src.run_config import RunConfig
from src.types import Bbox, TaxonScope


def a_config(**overrides: object) -> RunConfig:
    base: dict[str, object] = {
        "parquet_source_path": "test/sample-archive",
        "bounding_box": Bbox.from_coordinates(25.0, 47.0, -87.0, -66.0),
        "geocode_precision": 4,
        "taxon_scope": None,
        "limit_results": 1000,
        "sample_records": None,
        "max_taxa": None,
        "min_geocode_presence": None,
        "min_hex_records": None,
        "no_hex_floor": False,
        "terrestrial_only": False,
        "min_clusters_to_test": 2,
        "max_clusters_to_test": 20,
        "num_clusters_pinned": None,
        "metric_weights": {"silhouette": 0.7},
        "composition_metric": "betasim",
        "linkage": "ward",
        "reduction": "pcoa",
        "random_seed": 0,
        "hierarchy_levels": None,
        "default_display_level": 4,
        "log_file": "run.log",
        "no_images": False,
        "no_findings": False,
        "findings_output": "output/findings.html",
    }
    base.update(overrides)
    return RunConfig(**base)  # type: ignore[arg-type]


class TestRunConfig(unittest.TestCase):
    def test_a_run_cannot_rewrite_its_own_settings(self) -> None:
        """Frozen: the configuration is decided once, before any stage reads it."""
        config = a_config()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            config.geocode_precision = 5  # type: ignore[misc]

    def test_every_setting_appears_in_the_inputs_table(self) -> None:
        """The hand-written table omitted five settings, including the one that
        decides which cut gets published."""
        config = a_config()
        reported = {row["variable"] for row in config.as_rows()}
        self.assertEqual(reported, set(config.__dataclass_fields__))
        for omitted in (
            "sample_records",
            "default_display_level",
            "no_images",
            "no_findings",
            "findings_output",
        ):
            self.assertIn(omitted, reported)

    def test_the_table_renders_values_the_ui_can_show(self) -> None:
        """mo.ui.table wants scalars; a dict has to be stringified."""
        rows = {r["variable"]: r["value"] for r in a_config().as_rows()}
        self.assertIsInstance(rows["metric_weights"], str)
        self.assertEqual(rows["taxon_scope"], "(all taxa)")

    def test_a_scope_is_reported_as_itself(self) -> None:
        rows = {
            r["variable"]: r["value"]
            for r in a_config(taxon_scope=TaxonScope("class", "Aves")).as_rows()
        }
        self.assertEqual(str(rows["taxon_scope"]), "class:Aves")
