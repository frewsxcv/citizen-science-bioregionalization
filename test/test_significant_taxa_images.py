import unittest
from unittest import mock

import polars as pl

from src.dataframes.significant_taxa_images import (
    build_significant_taxa_images_df,
    canonical_name,
)


def _significant_differences_df() -> pl.DataFrame:
    return pl.DataFrame(
        {"cluster": [1, 2], "taxonId": [0, 1]},
        schema={"cluster": pl.UInt32, "taxonId": pl.UInt32},
    )


def _taxonomy_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "taxonId": [0, 1],
            "scientificName": [
                "Falco peregrinus Tunstall, 1771",
                "Cybianthus marginatus (Benth.) Pipoly",
            ],
            "gbifTaxonId": ["3DTGL", "9WQ2R"],
        },
        schema={
            "taxonId": pl.UInt32,
            "scientificName": pl.String,
            "gbifTaxonId": pl.String,
        },
    )


class TestCanonicalName(unittest.TestCase):
    def test_strips_author_and_year(self):
        self.assertEqual(
            canonical_name("Falco peregrinus Tunstall, 1771"), "Falco peregrinus"
        )

    def test_strips_parenthesised_author(self):
        self.assertEqual(
            canonical_name("Cybianthus marginatus (Benth.) Pipoly"),
            "Cybianthus marginatus",
        )

    def test_handles_genus_only_names(self):
        self.assertEqual(canonical_name("Bacopa Aubl."), "Bacopa")

    def test_keeps_infraspecific_epithets(self):
        self.assertEqual(
            canonical_name("Canis lupus familiaris Linnaeus"), "Canis lupus familiaris"
        )

    def test_leaves_a_bare_name_untouched(self):
        self.assertEqual(canonical_name("Panthera leo"), "Panthera leo")

    def test_handles_empty_input(self):
        self.assertEqual(canonical_name(""), "")


class TestBuildSignificantTaxaImagesDf(unittest.TestCase):
    def test_disabled_lookup_makes_no_network_call(self):
        with mock.patch(
            "src.dataframes.significant_taxa_images._fetch_wikidata_images"
        ) as fetch:
            result = build_significant_taxa_images_df(
                _significant_differences_df(), _taxonomy_df(), fetch_images=False
            )

        fetch.assert_not_called()
        self.assertEqual(result.columns, ["taxonId", "image_url"])
        self.assertEqual(result["image_url"].null_count(), result.height)

    def test_queries_wikidata_with_authorless_names(self):
        with mock.patch(
            "src.dataframes.significant_taxa_images._fetch_wikidata_images",
            return_value={},
        ) as fetch:
            build_significant_taxa_images_df(
                _significant_differences_df(), _taxonomy_df()
            )

        requested = sorted(fetch.call_args.args[0])
        self.assertEqual(
            requested, ["Cybianthus marginatus", "Falco peregrinus"]
        )

    def test_joins_images_back_onto_taxon_ids(self):
        with mock.patch(
            "src.dataframes.significant_taxa_images._fetch_wikidata_images",
            return_value={"Falco peregrinus": "https://example.invalid/falcon.jpg"},
        ):
            result = build_significant_taxa_images_df(
                _significant_differences_df(), _taxonomy_df()
            )

        by_taxon = dict(zip(result["taxonId"], result["image_url"]))
        self.assertEqual(by_taxon[0], "https://example.invalid/falcon.jpg")
        self.assertIsNone(by_taxon[1])

    def test_network_failure_degrades_to_no_images(self):
        # Images are decorative; a Wikidata outage must not fail the run.
        with mock.patch(
            "src.dataframes.significant_taxa_images.requests.post",
            side_effect=__import__("requests").exceptions.ConnectionError("boom"),
        ):
            result = build_significant_taxa_images_df(
                _significant_differences_df(), _taxonomy_df()
            )

        self.assertEqual(result["image_url"].null_count(), result.height)


if __name__ == "__main__":
    unittest.main()
