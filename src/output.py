"""
Output module for managing file paths and directory creation.

This module centralizes all output directory management and path standardization
to ensure consistent output file handling across the project. It also includes
functionality for writing JSON output files.
"""

import polars as pl
import logging
import os
from typing import Optional


import bioregion_rs

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


def write_json_output(
    cluster_significant_differences_df: pl.DataFrame,
    cluster_boundary_df: pl.DataFrame,
    taxonomy_df: pl.DataFrame,
    cluster_color_df: pl.DataFrame,
    significant_taxa_images_df: pl.DataFrame,
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
    json_str = bioregion_rs.build_json_output(
        cluster_significant_differences_df,
        cluster_boundary_df,
        taxonomy_df,
        cluster_color_df,
        significant_taxa_images_df,
    )

    output_file = prepare_file_path(output_path)

    with open(output_file, "w") as json_writer:
        json_writer.write(json_str)
