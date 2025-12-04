"""Utility functions for working with Darwin Core data."""

import polars as pl


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
