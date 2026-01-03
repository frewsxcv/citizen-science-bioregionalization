"""Command-line argument parser for citizen science bioregionalization."""

import argparse
from typing import Any


def create_argument_parser(
    geocode_precision: int,
    taxon_filter: str,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    limit_results: None | int,
    parquet_source_path: str,
    log_file: str | None,
    min_clusters_to_test: int,
    max_clusters_to_test: int,
    max_taxa: int | None,
    min_geocode_presence: float | None,
) -> argparse.ArgumentParser:
    """
    Create and configure the CLI argument parser.

    Args:
        geocode_precision: Default precision of the geocode
        taxon_filter: Default taxon filter (e.g., 'Aves')
        min_lat: Default minimum latitude for bounding box
        max_lat: Default maximum latitude for bounding box
        min_lon: Default minimum longitude for bounding box
        max_lon: Default maximum longitude for bounding box
        limit_results: Default limit for number of results fetched
        parquet_source_path: Default path to the parquet data source
        log_file: Default path to the log file
        min_clusters_to_test: Minimum number of clusters to test during optimization
        max_clusters_to_test: Maximum number of clusters to test during optimization
        max_taxa: Keep only top N taxa by total occurrence count
        min_geocode_presence: Keep only taxa present in at least this fraction of geocodes

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="Process Darwin Core CSV data and generate clusters."
    )

    # Add required options
    parser.add_argument(
        "--geocode-precision",
        type=int,
        help="Precision of the geocode",
        default=geocode_precision,
    )

    parser.add_argument(
        "--min-clusters",
        type=int,
        default=min_clusters_to_test,
        help="Minimum number of clusters to test during optimization",
    )
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=max_clusters_to_test,
        help="Maximum number of clusters to test during optimization",
    )
    parser.add_argument(
        "--log-file", type=str, help="Path to the log file", default=log_file
    )
    parser.add_argument(
        "--no-stop",
        action="store_true",
        default=False,
        help="Bypass mo.stop when running from the command line",
    )

    # Taxa filtering arguments
    parser.add_argument(
        "--max-taxa",
        type=int,
        default=max_taxa,
        help="Keep only top N taxa by total occurrence count. Recommended: 5000-10000 for large datasets.",
    )
    parser.add_argument(
        "--min-geocode-presence",
        type=float,
        default=min_geocode_presence,
        help="Keep only taxa present in at least this fraction of geocodes (0.0-1.0). Recommended: 0.02-0.05.",
    )

    # Add optional arguments
    parser.add_argument(
        "--taxon-filter",
        type=str,
        default=taxon_filter,
        help="Filter to a specific taxon (e.g., 'Aves')",
    )
    parser.add_argument(
        "--min-lat",
        type=float,
        default=min_lat,
        help="Minimum latitude for bounding box",
    )
    parser.add_argument(
        "--max-lat",
        type=float,
        default=max_lat,
        help="Maximum latitude for bounding box",
    )
    parser.add_argument(
        "--min-lon",
        type=float,
        default=min_lon,
        help="Minimum longitude for bounding box",
    )
    parser.add_argument(
        "--max-lon",
        type=float,
        default=max_lon,
        help="Maximum longitude for bounding box",
    )
    parser.add_argument(
        "--limit-results",
        type=int,
        default=limit_results,
        help="Limit the number of results fetched",
    )

    # Positional arguments
    parser.add_argument(
        "parquet_source_path",
        type=str,
        nargs="?",
        help="Path to the parquet data source",
        default=parquet_source_path,
    )

    return parser


def parse_args_with_defaults(
    geocode_precision: int,
    taxon_filter: str,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    limit_results: None | int,
    parquet_source_path: str,
    log_file: str | None,
    min_clusters_to_test: int = 2,
    max_clusters_to_test: int = 20,
    max_taxa: int | None = None,
    min_geocode_presence: float | None = None,
) -> Any:
    """
    Create argument parser and parse command-line arguments.

    Args:
        geocode_precision: Default precision of the geocode
        taxon_filter: Default taxon filter (e.g., 'Aves')
        min_lat: Default minimum latitude for bounding box
        max_lat: Default maximum latitude for bounding box
        min_lon: Default minimum longitude for bounding box
        max_lon: Default maximum longitude for bounding box
        limit_results: Default limit for number of results fetched
        parquet_source_path: Default path to the parquet data source
        log_file: Default path to the log file
        min_clusters_to_test: Minimum number of clusters to test during optimization
        max_clusters_to_test: Maximum number of clusters to test during optimization
        max_taxa: Keep only top N taxa by total occurrence count
        min_geocode_presence: Keep only taxa present in at least this fraction of geocodes

    Returns:
        Parsed command-line arguments
    """
    parser = create_argument_parser(
        geocode_precision=geocode_precision,
        taxon_filter=taxon_filter,
        min_lat=min_lat,
        max_lat=max_lat,
        min_lon=min_lon,
        max_lon=max_lon,
        limit_results=limit_results,
        parquet_source_path=parquet_source_path,
        log_file=log_file,
        min_clusters_to_test=min_clusters_to_test,
        max_clusters_to_test=max_clusters_to_test,
        max_taxa=max_taxa,
        min_geocode_presence=min_geocode_presence,
    )
    return parser.parse_args()
