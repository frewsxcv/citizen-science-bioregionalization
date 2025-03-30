"""
Output module for managing file paths and directory creation.

This module centralizes all output directory management and path standardization
to ensure consistent output file handling across the project.
"""

import os
import logging
from typing import Optional

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
