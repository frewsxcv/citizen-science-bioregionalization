import logging
from typing import Dict, List

import dataframely as dy
import polars as pl
import requests

logger = logging.getLogger(__name__)

from src.dataframes.cluster_significant_differences import (
    ClusterSignificantDifferencesSchema,
)
from src.dataframes.taxonomy import TaxonomySchema


def _fetch_wikidata_images(gbif_taxon_ids: List[int]) -> Dict[int, str]:
    """
    Fetches image URLs from Wikidata for a list of GBIF taxon IDs.

    Args:
        gbif_taxon_ids: List of GBIF taxon IDs as integers

    Returns:
        Dictionary mapping GBIF taxon ID (int) to image URL (str)
    """
    if not gbif_taxon_ids:
        return {}

    # Convert integers to strings for SPARQL query
    gbif_ids_str = " ".join([f'"{id}"' for id in gbif_taxon_ids])
    sparql_query = f"""
        SELECT ?gbif_taxon_id (SAMPLE(?image) AS ?image) WHERE {{
            VALUES ?gbif_taxon_id {{ {gbif_ids_str} }} .
            ?item wdt:P846 ?gbif_taxon_id .
            OPTIONAL {{ ?item wdt:P18 ?image }} .
        }} GROUP BY ?gbif_taxon_id
    """

    endpoint = "https://query.wikidata.org/sparql"
    data = {"query": sparql_query, "format": "json"}
    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": "CitizenScienceBioregionalization/1.0",
    }

    try:
        response = requests.post(endpoint, data=data, headers=headers)
        response.raise_for_status()
        results = response.json()
        image_map: Dict[int, str] = {}
        for binding in results["results"]["bindings"]:
            if "image" in binding:
                # Convert string response back to int
                gbif_id = int(binding["gbif_taxon_id"]["value"])
                image_map[gbif_id] = binding["image"]["value"]
        return image_map
    except requests.exceptions.RequestException as e:
        print(f"Error fetching from Wikidata: {e}")
        return {}


class SignificantTaxaImagesSchema(dy.Schema):
    taxonId = dy.UInt32(nullable=False)
    image_url = dy.String(nullable=True)


def build_significant_taxa_images_df(
    cluster_significant_differences_df: dy.DataFrame[
        ClusterSignificantDifferencesSchema
    ],
    taxonomy_df: dy.DataFrame[TaxonomySchema],
) -> dy.DataFrame[SignificantTaxaImagesSchema]:
    """Build a SignificantTaxaImagesSchema DataFrame with image URLs from Wikidata.

    Fetches image URLs from Wikidata for taxa that have significant differences
    between clusters.

    Args:
        cluster_significant_differences_df: DataFrame of significant taxa differences
        taxonomy_df: DataFrame of taxonomy information with GBIF taxon IDs

    Returns:
        A validated DataFrame conforming to SignificantTaxaImagesSchema
    """
    logger.info("build_significant_taxa_images_df: Starting")

    significant_taxa_df = cluster_significant_differences_df.select("taxonId").unique()

    significant_taxa_with_gbif = significant_taxa_df.join(
        taxonomy_df.select(["taxonId", "gbifTaxonId"]), on="taxonId"
    )

    gbif_ids = significant_taxa_with_gbif.get_column("gbifTaxonId").unique().to_list()

    image_map = _fetch_wikidata_images(gbif_ids)

    if not image_map:
        return SignificantTaxaImagesSchema.validate(
            significant_taxa_df.with_columns(
                image_url=pl.lit(None, dtype=pl.String)
            ).select(["taxonId", "image_url"])
        )

    images_df = pl.DataFrame(
        {
            "gbifTaxonId": list(image_map.keys()),
            "image_url": list(image_map.values()),
        }
    )

    # Join images back to the significant taxa with gbifTaxonId
    result_df = significant_taxa_with_gbif.join(
        images_df, on="gbifTaxonId", how="left"
    ).select(["taxonId", "image_url"])

    return SignificantTaxaImagesSchema.validate(result_df)
