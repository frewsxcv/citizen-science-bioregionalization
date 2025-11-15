import dataframely as dy
import polars as pl
import shapely
import json
import requests
from src import output
from src.dataframes.cluster_significant_differences import (
    ClusterSignificantDifferencesSchema,
)
from src.dataframes.cluster_boundary import ClusterBoundarySchema
from src.dataframes.taxonomy import TaxonomySchema
from src.dataframes.cluster_color import ClusterColorSchema


def _fetch_wikidata_images(gbif_taxon_ids: list[int]) -> dict[int, str]:
    """
    Fetches image URLs from Wikidata for a list of GBIF taxon IDs.
    """
    if not gbif_taxon_ids:
        return {}

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
    headers = {"Accept": "application/sparql-results+json", "User-Agent": "CitizenScienceBioregionalization/1.0"}

    try:
        response = requests.post(endpoint, data=data, headers=headers)
        response.raise_for_status()
        results = response.json()
        image_map = {}
        for binding in results["results"]["bindings"]:
            if "image" in binding:
                gbif_id = int(binding["gbif_taxon_id"]["value"])
                image_map[gbif_id] = binding["image"]["value"]
        return image_map
    except requests.exceptions.RequestException as e:
        print(f"Error fetching from Wikidata: {e}")
        return {}


def _wkb_to_geojson(wkb: bytes) -> dict:
    """
    Convert WKB geometry to a GeoJSON-compatible dictionary.
    """
    geom = shapely.from_wkb(wkb)
    return json.loads(shapely.to_geojson(geom))


def write_json_output(
    cluster_significant_differences_df: dy.DataFrame[ClusterSignificantDifferencesSchema],
    cluster_boundary_df: dy.DataFrame[ClusterBoundarySchema],
    taxonomy_df: dy.DataFrame[TaxonomySchema],
    cluster_color_df: dy.DataFrame[ClusterColorSchema],
    output_path: str,
) -> None:
    """
    Writes the cluster data to a JSON file.

    Args:
        cluster_significant_differences_df: DataFrame with significant taxa for each cluster.
        cluster_boundary_df: DataFrame with the boundary for each cluster.
        taxonomy_df: DataFrame with taxonomy information.
        cluster_color_df: DataFrame with color information for each cluster.
        output_path: The path to write the JSON file to.
    """
    output_data = []

    cluster_data_df = cluster_boundary_df.join(cluster_color_df, on="cluster")

    for row in cluster_data_df.iter_rows(named=True):
        cluster_id = row["cluster"]
        boundary_wkb = row["geometry"]
        color = row["color"]
        darkened_color = row["darkened_color"]

        significant_taxa_df = cluster_significant_differences_df.filter(
            pl.col("cluster") == cluster_id
        ).join(taxonomy_df, on="taxonId")

        significant_taxa = []
        gbif_ids = [
            r["gbifTaxonId"]
            for r in significant_taxa_df.iter_rows(named=True)
            if r["gbifTaxonId"]
        ]
        image_map = _fetch_wikidata_images(gbif_ids)

        for r in significant_taxa_df.iter_rows(named=True):
            gbif_taxon_id = r["gbifTaxonId"]
            significant_taxa.append(
                {
                    "scientific_name": r["scientificName"],
                    "gbif_taxon_id": gbif_taxon_id,
                    "p_value": r["p_value"],
                    "log2_fold_change": r["log2_fold_change"],
                    "cluster_count": r["cluster_count"],
                    "neighbor_count": r["neighbor_count"],
                    "image_url": image_map.get(gbif_taxon_id),
                }
            )

        output_data.append(
            {
                "cluster": cluster_id,
                "boundary": _wkb_to_geojson(boundary_wkb),
                "significant_taxa": significant_taxa,
                "color": color,
                "darkened_color": darkened_color,
            }
        )

    # Prepare the output file path
    output_file = output.prepare_file_path(output_path)

    with open(output_file, "w") as json_writer:
        json.dump(output_data, json_writer, indent=2)
