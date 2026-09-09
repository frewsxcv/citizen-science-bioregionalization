import logging

import polars as pl
import requests

logger = logging.getLogger(__name__)

WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
REQUEST_TIMEOUT_SECONDS = 60


def canonical_name(scientific_name: str) -> str:
    """Strip the authorship from a Darwin Core scientific name.

    GBIF's `scientificName` carries the author and year ("Falco peregrinus
    Tunstall, 1771"), while Wikidata's taxon name (P225) holds the bare name
    ("Falco peregrinus"), so the two only match once authorship is removed.

    The name part is the leading capitalised token followed by any purely
    lowercase alphabetic tokens (the specific and infraspecific epithets).
    Authorship always begins with something else — a capital, a parenthesis or a
    digit — so the first such token ends the name:

        "Falco peregrinus Tunstall, 1771"     -> "Falco peregrinus"
        "Cybianthus marginatus (Benth.) Pipoly" -> "Cybianthus marginatus"
        "Bacopa Aubl."                        -> "Bacopa"
    """
    tokens = scientific_name.split()
    if not tokens:
        return ""
    name = [tokens[0]]
    for token in tokens[1:]:
        if token.isalpha() and token.islower():
            name.append(token)
        else:
            break
    return " ".join(name)


def _fetch_wikidata_images(scientific_names: list[str]) -> dict[str, str]:
    """Fetch image URLs from Wikidata for a list of canonical taxon names.

    Matches on taxon name (P225) rather than GBIF taxon id (P846): GBIF's keys
    are now alphanumeric and no longer correspond to the numeric ids Wikidata
    recorded, so a P846 lookup either finds nothing or — worse, when keys happen
    to be numeric — silently resolves to an unrelated taxon.

    Args:
        scientific_names: Canonical taxon names, without authorship.

    Returns:
        Mapping from canonical name to image URL, omitting taxa with no image.
    """
    if not scientific_names:
        return {}

    values = " ".join(f'"{name}"' for name in scientific_names)
    sparql_query = f"""
        SELECT ?taxon_name (SAMPLE(?image) AS ?image) WHERE {{
            VALUES ?taxon_name {{ {values} }} .
            ?item wdt:P225 ?taxon_name .
            OPTIONAL {{ ?item wdt:P18 ?image }} .
        }} GROUP BY ?taxon_name
    """

    data = {"query": sparql_query, "format": "json"}
    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": "CitizenScienceBioregionalization/1.0",
    }

    try:
        response = requests.post(
            WIKIDATA_ENDPOINT,
            data=data,
            headers=headers,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        results = response.json()
    except (requests.exceptions.RequestException, ValueError) as e:
        # Images are decorative; a failed lookup must not fail the run.
        logger.warning("Wikidata image lookup failed: %s", e)
        return {}

    return {
        binding["taxon_name"]["value"]: binding["image"]["value"]
        for binding in results["results"]["bindings"]
        if "image" in binding
    }


def build_significant_taxa_images_df(
    cluster_significant_differences_df: pl.DataFrame,
    taxonomy_df: pl.DataFrame,
    fetch_images: bool = True,
) -> pl.DataFrame:
    """Attach a Wikidata image URL to each significant taxon, where one exists.

    Args:
        cluster_significant_differences_df: Significant taxa per cluster.
        taxonomy_df: Taxonomy, providing `scientificName` per `taxonId`.
        fetch_images: When False, skip the network call and return null image
            URLs. Keeps runs offline and reproducible.

    Returns:
        One row per significant taxon, with `taxonId` and a nullable `image_url`.
    """
    logger.info("build_significant_taxa_images_df: Starting")

    significant_taxa_df = cluster_significant_differences_df.select("taxonId").unique()

    def _without_images() -> pl.DataFrame:
        return significant_taxa_df.with_columns(
            image_url=pl.lit(None, dtype=pl.String)
        ).select(["taxonId", "image_url"])

    if not fetch_images:
        logger.info("build_significant_taxa_images_df: image lookup disabled")
        return _without_images()

    named = significant_taxa_df.join(
        taxonomy_df.select(["taxonId", "scientificName"]), on="taxonId"
    ).with_columns(
        pl.col("scientificName")
        .map_elements(canonical_name, return_dtype=pl.String)
        .alias("canonicalName")
    )

    image_map = _fetch_wikidata_images(
        named.get_column("canonicalName").unique().drop_nulls().to_list()
    )
    logger.info(
        "build_significant_taxa_images_df: %d of %d taxa have images",
        len(image_map),
        named.height,
    )

    if not image_map:
        return _without_images()

    images_df = pl.DataFrame(
        {
            "canonicalName": list(image_map.keys()),
            "image_url": list(image_map.values()),
        }
    )

    return named.join(images_df, on="canonicalName", how="left").select(
        ["taxonId", "image_url"]
    )
