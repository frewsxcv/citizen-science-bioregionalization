"""Utility functions for working with Darwin Core data."""

from pathlib import Path

import polars as pl

from src.dataframes.darwin_core import scan_darwin_core_archive


def get_parquet_to_darwin_core_column_mapping() -> dict[str, str]:
    """
    Get the mapping from lowercase column names (used in parquet snapshots)
    to camelCase column names (Darwin Core standard).

    Returns:
        Dictionary mapping lowercase column names to camelCase column names
    """
    return {
        "decimallatitude": "decimalLatitude",
        "decimallongitude": "decimalLongitude",
        "taxonkey": "taxonKey",
        "specieskey": "speciesKey",
        "acceptedtaxonkey": "acceptedTaxonKey",
        "kingdomkey": "kingdomKey",
        "phylumkey": "phylumKey",
        "classkey": "classKey",
        "orderkey": "orderKey",
        "familykey": "familyKey",
        "genuskey": "genusKey",
        "subgenuskey": "subgenusKey",
        "taxonrank": "taxonRank",
        "scientificname": "scientificName",
        "verbatimscientificname": "verbatimScientificName",
        "countrycode": "countryCode",
        "gbifid": "gbifID",
        "datasetkey": "datasetKey",
        "occurrenceid": "occurrenceID",
        "eventdate": "eventDate",
        "basisofrecord": "basisOfRecord",
        "individualcount": "individualCount",
        "publishingorgkey": "publishingOrgKey",
        "coordinateuncertaintyinmeters": "coordinateUncertaintyInMeters",
        "coordinateprecision": "coordinatePrecision",
        "hascoordinate": "hasCoordinate",
        "hasgeospatialissues": "hasGeospatialIssues",
        "stateprovince": "stateProvince",
        "iucnredlistcategory": "iucnRedListCategory",
    }


def rename_parquet_columns_to_darwin_core(
    lazy_frame: pl.LazyFrame,
) -> pl.LazyFrame:
    """
    Rename columns from lowercase (parquet snapshots) to camelCase (Darwin Core standard).

    Only renames columns that actually exist in the LazyFrame schema.

    Args:
        lazy_frame: The Polars LazyFrame with lowercase column names

    Returns:
        LazyFrame with columns renamed to Darwin Core camelCase standard
    """
    parquet_to_darwin_core_columns = get_parquet_to_darwin_core_column_mapping()
    existing_columns = set(lazy_frame.collect_schema().names())
    columns_to_rename = {
        k: v for k, v in parquet_to_darwin_core_columns.items() if k in existing_columns
    }
    return lazy_frame.rename(columns_to_rename)


def build_taxon_filter(taxon_name: str) -> pl.Expr:
    """
    Build a Polars expression to filter observations by taxon name.

    Checks if the taxon name matches any taxonomic rank column:
    kingdom, phylum, class, order, family, genus, or species.

    Args:
        taxon_name: The taxon name to filter by

    Returns:
        A Polars expression that matches observations where any taxonomic
        rank column equals the given taxon name
    """
    return (
        (pl.col("kingdom") == taxon_name)
        | (pl.col("phylum") == taxon_name)
        | (pl.col("class") == taxon_name)
        | (pl.col("order") == taxon_name)
        | (pl.col("family") == taxon_name)
        | (pl.col("genus") == taxon_name)
        | (pl.col("species") == taxon_name)
    )


def load_darwin_core_data(
    source_path: str,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    limit_results: int | None,
    taxon_filter: str = "",
) -> pl.LazyFrame:
    """
    Load Darwin Core data from either a Darwin Core archive or Parquet file.

    Automatically detects the source type and applies geographic and taxonomic filters.

    Args:
        source_path: Path to either a Darwin Core archive directory or Parquet file/directory
        min_lat: Minimum latitude for bounding box filter
        max_lat: Maximum latitude for bounding box filter
        min_lon: Minimum longitude for bounding box filter
        max_lon: Maximum longitude for bounding box filter
        limit_results: Maximum number of results to return
        taxon_filter: Optional taxon name to filter by (e.g., 'Aves')

    Returns:
        A LazyFrame with filtered Darwin Core data
    """
    # Detect if source is a Darwin Core archive (directory with meta.xml) or parquet
    path = Path(source_path)
    is_darwin_core_archive = path.is_dir() and (path / "meta.xml").exists()

    # Build base filters for geographic bounds
    # Use camelCase column names (Darwin Core standard)
    # First filter out null coordinates, then apply bounds
    base_filters = (
        pl.col("decimalLatitude").is_not_null()
        & pl.col("decimalLongitude").is_not_null()
        & pl.col("decimalLatitude").is_between(min_lat, max_lat)
        & pl.col("decimalLongitude").is_between(min_lon, max_lon)
    )

    # Add taxon filter if specified
    if taxon_filter:
        taxon_filter_expr = build_taxon_filter(taxon_filter)
        base_filters = base_filters & taxon_filter_expr

    if is_darwin_core_archive:
        # Load from Darwin Core archive (already uses camelCase)
        inner_lf = scan_darwin_core_archive(source_path)
    else:
        # Load from parquet snapshot and rename columns to camelCase
        # Public GCS buckets (like GBIF) are accessible without credentials
        inner_lf = pl.scan_parquet(source_path)
        inner_lf = rename_parquet_columns_to_darwin_core(inner_lf)

    # Apply filters and limit to the lazy frame
    inner_lf = inner_lf.filter(base_filters)
    if limit_results is not None:
        inner_lf = inner_lf.limit(limit_results)

    return inner_lf
