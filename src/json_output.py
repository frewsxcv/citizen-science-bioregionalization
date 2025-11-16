import json

import dataframely as dy
import polars as pl
import shapely

from src import output
from src.dataframes.cluster_boundary import ClusterBoundarySchema
from src.dataframes.cluster_color import ClusterColorSchema
from src.dataframes.cluster_significant_differences import (
    ClusterSignificantDifferencesSchema,
)
from src.dataframes.significant_taxa_images import SignificantTaxaImagesSchema
from src.dataframes.taxonomy import TaxonomySchema


def _wkb_to_geojson(wkb: bytes) -> dict:
    """
    Convert WKB geometry to a GeoJSON-compatible dictionary.
    """
    geom = shapely.from_wkb(wkb)
    return json.loads(shapely.to_geojson(geom))


def write_json_output(
    cluster_significant_differences_df: dy.DataFrame[
        ClusterSignificantDifferencesSchema
    ],
    cluster_boundary_df: dy.DataFrame[ClusterBoundarySchema],
    taxonomy_df: dy.DataFrame[TaxonomySchema],
    cluster_color_df: dy.DataFrame[ClusterColorSchema],
    significant_taxa_images_df: dy.DataFrame[SignificantTaxaImagesSchema],
    output_path: str,
) -> None:
    """
    Writes the cluster data to a JSON file.

    Args:
        cluster_significant_differences_df: DataFrame with significant taxa for each cluster.
        cluster_boundary_df: DataFrame with the boundary for each cluster.
        taxonomy_df: DataFrame with taxonomy information.
        cluster_color_df: DataFrame with color information for each cluster.
        significant_taxa_images_df: DataFrame with image URLs for significant taxa.
        output_path: The path to write the JSON file to.
    """
    output_data = []

    cluster_data_df = cluster_boundary_df.join(cluster_color_df, on="cluster")

    for row in cluster_data_df.iter_rows(named=True):
        cluster_id = row["cluster"]
        boundary_wkb = row["geometry"]
        color = row["color"]
        darkened_color = row["darkened_color"]

        significant_taxa_df = (
            cluster_significant_differences_df.filter(pl.col("cluster") == cluster_id)
            .join(taxonomy_df, on="taxonId")
            .join(significant_taxa_images_df, on="taxonId", how="left")
        )

        significant_taxa = []
        for r in significant_taxa_df.iter_rows(named=True):
            significant_taxa.append(
                {
                    "scientific_name": r["scientificName"],
                    "gbif_taxon_id": r["gbifTaxonId"],
                    "taxon_id": r["taxonId"],
                    "log2_fold_change": r["log2_fold_change"],
                    "cluster_count": r["cluster_count"],
                    "neighbor_count": r["neighbor_count"],
                    "high_log2_high_count_score": r["high_log2_high_count_score"],
                    "low_log2_high_count_score": r["low_log2_high_count_score"],
                    "image_url": r["image_url"],
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
