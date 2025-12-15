"""
Output module for managing file paths and directory creation.

This module centralizes all output directory management and path standardization
to ensure consistent output file handling across the project. It also includes
functionality for writing JSON output files.
"""

import json
import logging
import os
from typing import Optional

import dataframely as dy
import polars as pl
import shapely

from src.dataframes.cluster_boundary import ClusterBoundarySchema
from src.dataframes.cluster_color import ClusterColorSchema
from src.dataframes.cluster_significant_differences import (
    ClusterSignificantDifferencesSchema,
)
from src.dataframes.significant_taxa_images import SignificantTaxaImagesSchema
from src.dataframes.taxonomy import TaxonomySchema

# Default output directory
OUTPUT_DIR = "output"

# Fixed output filenames
GEOJSON_FILENAME = "output.geojson"
HTML_FILENAME = "output.html"


def ensure_output_dir() -> None:
    """
    Create the output directory if it doesn't exist.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_output_path(filename: str) -> str:
    """
    Gets the full path for an output file in the output directory.

    Args:
        filename: The filename to place in the output directory

    Returns:
        The full path to the file in the output directory
    """
    ensure_output_dir()
    return os.path.join(OUTPUT_DIR, filename)


def normalize_path(path: str) -> str:
    """
    Normalize a path to ensure it's in the output directory if it doesn't have a directory component.

    Args:
        path: The path to normalize

    Returns:
        The normalized path
    """
    if not path.startswith(f"{OUTPUT_DIR}/") and not os.path.dirname(path):
        return os.path.join(OUTPUT_DIR, path)
    return path


def get_geojson_path() -> str:
    """
    Get the standard path for the GeoJSON output file.

    Returns:
        Path to the GeoJSON output file
    """
    return get_output_path(GEOJSON_FILENAME)


def get_html_path() -> str:
    """
    Get the standard path for the HTML output file.

    Returns:
        Path to the HTML output file
    """
    return get_output_path(HTML_FILENAME)


def prepare_file_path(path: str) -> str:
    """
    Prepare a file path for writing by ensuring its directory exists.

    Args:
        path: The path to prepare

    Returns:
        The same path after ensuring its directory exists
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    return path


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
    output_file = prepare_file_path(output_path)

    with open(output_file, "w") as json_writer:
        json.dump(output_data, json_writer, indent=2)
