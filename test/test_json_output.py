"""Tests for the JSON document handed to the frontend.

`write_json_output` delegates to `bioregion_rs.build_json_output`, which reads
each column at a fixed dtype. Nothing else in the suite exercises that boundary,
so a dtype change on the Python side (such as GBIF taxon keys becoming strings)
would otherwise only surface in the full-pipeline notebook test.
"""

import json
import tempfile
import unittest
from pathlib import Path

import polars as pl

from src.output import write_json_output
from test.fixtures.cluster_boundary import mock_cluster_boundary_df
from test.fixtures.cluster_color import mock_cluster_color_df


def _mock_significant_differences_df() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "cluster": 1,
                "taxonId": 0,
                "log2_fold_change": 2.5,
                "cluster_count": 10,
                "neighbor_count": 3,
                "high_log2_high_count_score": 0.8,
                "low_log2_high_count_score": 0.1,
            },
            {
                "cluster": 2,
                "taxonId": 1,
                "log2_fold_change": -1.25,
                "cluster_count": 4,
                "neighbor_count": 9,
                "high_log2_high_count_score": 0.2,
                "low_log2_high_count_score": 0.7,
            },
        ]
    ).with_columns(
        pl.col("cluster").cast(pl.UInt32),
        pl.col("taxonId").cast(pl.UInt32),
        pl.col("cluster_count").cast(pl.UInt32),
        pl.col("neighbor_count").cast(pl.UInt32),
    )


def _mock_taxonomy_df() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {"taxonId": 0, "scientificName": "Panthera leo", "gbifTaxonId": "5219404"},
            # Alphanumeric, as current GBIF snapshots emit.
            {"taxonId": 1, "scientificName": "Canis lupus", "gbifTaxonId": "3DTGL"},
        ]
    ).with_columns(
        pl.col("taxonId").cast(pl.UInt32),
        pl.col("gbifTaxonId").cast(pl.String),
    )


def _mock_images_df(url: str | None = None) -> pl.DataFrame:
    return pl.DataFrame(
        {"taxonId": [0, 1], "image_url": [url, None]},
        schema={"taxonId": pl.UInt32, "image_url": pl.String},
    )


class TestWriteJsonOutput(unittest.TestCase):
    def _write_and_read(self, **overrides) -> list:
        args = {
            "cluster_significant_differences_df": _mock_significant_differences_df(),
            "cluster_boundary_df": mock_cluster_boundary_df(),
            "taxonomy_df": _mock_taxonomy_df(),
            "cluster_color_df": mock_cluster_color_df(),
            "significant_taxa_images_df": _mock_images_df(),
        }
        args.update(overrides)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "aggregations.json"
            write_json_output(output_path=str(path), **args)
            return json.loads(path.read_text())

    def test_emits_one_entry_per_cluster(self):
        result = self._write_and_read()

        self.assertEqual({entry["cluster"] for entry in result}, {1, 2})

    def test_alphanumeric_taxon_keys_survive_as_strings(self):
        result = self._write_and_read()

        keys = {
            taxon["gbif_taxon_id"]
            for entry in result
            for taxon in entry["significant_taxa"]
        }
        self.assertEqual(keys, {"5219404", "3DTGL"})

    def test_taxonomy_and_boundary_are_joined_in(self):
        result = self._write_and_read()

        by_cluster = {entry["cluster"]: entry for entry in result}
        self.assertEqual(
            by_cluster[1]["significant_taxa"][0]["scientific_name"], "Panthera leo"
        )
        self.assertEqual(by_cluster[1]["color"], "#ff0000")
        self.assertIn("boundary", by_cluster[1])

    def test_image_urls_are_carried_through(self):
        result = self._write_and_read(
            significant_taxa_images_df=_mock_images_df("https://example.invalid/lion.jpg")
        )

        by_cluster = {entry["cluster"]: entry for entry in result}
        self.assertEqual(
            by_cluster[1]["significant_taxa"][0]["image_url"],
            "https://example.invalid/lion.jpg",
        )
        self.assertIsNone(by_cluster[2]["significant_taxa"][0]["image_url"])


if __name__ == "__main__":
    unittest.main()
