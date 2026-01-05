"""Utility functions for working with Darwin Core data."""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import polars as pl


@dataclass
class _Meta:
    """Metadata parsed from a Darwin Core archive's meta.xml file."""

    core_file: str
    has_header: bool
    separator: str
    columns: list[str]
    quote_char: str
    encoding: str
    default_fields: dict[str, str]


# Columns that should be cast to Categorical/Enum types after loading
# Maps column name (lowercase) to target data type
_CATEGORICAL_CASTS: dict[str, pl.DataType] = {}


# Actual categorical casting is done via cast_taxonomic_columns()
_BASE_SCHEMA: dict[str, pl.DataType] = {
    # Geographic fields
    "decimallatitude": pl.Float64(),
    "decimallongitude": pl.Float64(),
    # Taxonomic metadata
    "scientificname": pl.String(),
    "taxonkey": pl.UInt32(),
    # Observation metadata
    "individualcount": pl.Int32(),
}

# Lowercase to camelCase mapping for Darwin Core standard column names
_LOWER_TO_CAMEL: dict[str, str] = {
    "decimallatitude": "decimalLatitude",
    "decimallongitude": "decimalLongitude",
    "taxonkey": "taxonKey",
    "scientificname": "scientificName",
    "individualcount": "individualCount",
    "catalognumber": "catalogNumber",
    "hasgeospatialissues": "hasGeospatialIssues",
    "specieskey": "speciesKey",
    "acceptedtaxonkey": "acceptedTaxonKey",
    "kingdomkey": "kingdomKey",
    "phylumkey": "phylumKey",
    "classkey": "classKey",
    "orderkey": "orderKey",
    "familykey": "familyKey",
    "genuskey": "genusKey",
    "subgenuskey": "subgenusKey",
    "verbatimscientificname": "verbatimScientificName",
    "countrycode": "countryCode",
    "gbifid": "gbifID",
    "datasetkey": "datasetKey",
    "occurrenceid": "occurrenceID",
    "eventdate": "eventDate",
    "basisofrecord": "basisOfRecord",
    "publishingorgkey": "publishingOrgKey",
    "coordinateuncertaintyinmeters": "coordinateUncertaintyInMeters",
    "coordinateprecision": "coordinatePrecision",
    "stateprovince": "stateProvince",
    "iucnredlistcategory": "iucnRedListCategory",
}

# Derived schemas for external use
SCHEMA_LOWER: dict[str, pl.DataType] = _BASE_SCHEMA
SCHEMA_CAMEL: dict[str, pl.DataType] = {
    _LOWER_TO_CAMEL.get(k, k): v for k, v in _BASE_SCHEMA.items()
}


def cast_taxonomic_columns(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Cast taxonomic columns to their appropriate Categorical types.

    This function applies consistent casting for taxonomic hierarchy columns
    (phylum, class, order, family, genus) to optimize
    memory usage and query performance.

    Args:
        lf: Input LazyFrame with taxonomic columns

    Returns:
        LazyFrame with taxonomic columns cast to Categorical types
    """
    # Build a mapping from lowercase column name to actual column name
    actual_cols = lf.collect_schema().names()
    lower_to_actual: dict[str, str] = {col.lower(): col for col in actual_cols}

    cast_exprs = []
    for col_lower, target_type in _CATEGORICAL_CASTS.items():
        # Look up the actual column name (handles both lowercase and camelCase)
        actual_col = lower_to_actual.get(col_lower)
        if actual_col is not None:
            cast_exprs.append(pl.col(actual_col).cast(target_type))

    if cast_exprs:
        lf = lf.with_columns(cast_exprs)

    return lf


def _parse_meta(meta_path: Path) -> _Meta:
    """Parse the meta.xml file and return metadata for reading the archive.

    This method reads the Darwin Core archive's metafile (meta.xml) to extract
    parameters needed to correctly parse the core data file.

    Args:
        meta_path: Path to the meta.xml file

    Returns:
        A _Meta instance containing the parsed metadata.
    """
    tree = ET.parse(meta_path)
    root = tree.getroot()

    # Handle XML namespace if present
    ns = {"dwc": "http://rs.tdwg.org/dwc/text/"}

    # Try with namespace first, then without
    core_elem = root.find("dwc:core", ns)
    if core_elem is None:
        core_elem = root.find(".//core")
    if core_elem is None:
        raise ValueError("meta.xml does not contain <core> element")

    # file location – in <files><location>relative/path</location></files>
    files_elem = core_elem.find(".//files")
    if files_elem is None:
        files_elem = core_elem.find("dwc:files", ns)
    if files_elem is None:
        raise ValueError("<core> missing <files>")

    location_elem = files_elem.find(".//location")
    if location_elem is None:
        location_elem = files_elem.find("dwc:location", ns)
    if location_elem is None or not location_elem.text:
        raise ValueError("<files> missing <location>")
    core_file = location_elem.text.strip()

    # attributes from the <core> element, with defaults from the guide
    separator = core_elem.get("fieldsTerminatedBy", ",")
    if separator == "\\t":
        separator = "\t"

    quote_char = core_elem.get("fieldsEnclosedBy", '"')
    encoding = core_elem.get("encoding", "utf-8")
    ignore_header = int(core_elem.get("ignoreHeaderLines", "0"))
    has_header = ignore_header >= 1

    # fields and default values
    fields: list[str] = []
    default_fields: dict[str, str] = {}

    field_elems = core_elem.findall(".//field")
    if not field_elems:
        field_elems = core_elem.findall("dwc:field", ns)

    for field_elem in field_elems:
        term_uri = field_elem.get("term")
        if term_uri is None:
            continue

        term = term_uri.rsplit("/", 1)[-1].rsplit("#", 1)[-1]
        index_str = field_elem.get("index")

        if index_str is not None:
            try:
                idx = int(index_str)
            except ValueError:
                continue
            if len(fields) <= idx:
                fields.extend([""] * (idx - len(fields) + 1))
            fields[idx] = term
        else:
            default_value = field_elem.get("default")
            if default_value is not None:
                default_fields[term] = default_value

    # Handle <id index="0" /> element
    id_elem = core_elem.find(".//id")
    if id_elem is None:
        id_elem = core_elem.find("dwc:id", ns)

    if id_elem is not None:
        idx2 = id_elem.get("index")
        if idx2 is not None:
            try:
                idx = int(idx2)
                if len(fields) <= idx:
                    fields.extend([""] * (idx - len(fields) + 1))
                if not fields[idx]:
                    fields[idx] = "id"
            except (ValueError, IndexError):
                pass

    # fill any empty column names with fallback names
    final_fields = [name if name else f"col_{i}" for i, name in enumerate(fields)]

    return _Meta(
        core_file=core_file,
        has_header=has_header,
        separator=separator,
        columns=final_fields,
        quote_char=quote_char,
        encoding=encoding,
        default_fields=default_fields,
    )


def scan_darwin_core_archive(
    path: Union[str, Path], **scan_csv_kwargs: Any
) -> pl.LazyFrame:
    """Scan an unpacked Darwin Core Archive directory lazily.

    Args:
        path: Path to a directory containing meta.xml and the core data file.
        **scan_csv_kwargs: Extra keyword arguments forwarded to pl.scan_csv.

    Returns:
        A Polars LazyFrame containing the Darwin Core data.
    """
    base_dir = Path(path)
    meta_path = base_dir / "meta.xml"
    if not meta_path.exists():
        raise FileNotFoundError("meta.xml not found in archive directory")

    meta = _parse_meta(meta_path)
    data_path = base_dir / meta.core_file

    schema_from_meta = {
        col: SCHEMA_CAMEL[col] for col in meta.columns if col in SCHEMA_CAMEL
    }
    scan_csv_kwargs.setdefault("schema_overrides", {}).update(schema_from_meta)

    if meta.encoding.upper() != "UTF-8":
        raise NotImplementedError(
            f"Only UTF-8 encoding is supported, got {meta.encoding}"
        )

    inner = pl.scan_csv(
        data_path,
        separator=meta.separator,
        has_header=meta.has_header,
        new_columns=meta.columns if not meta.has_header else None,
        quote_char=meta.quote_char,
        low_memory=True,
        encoding="utf8",
        cache=False,
        **scan_csv_kwargs,
    )

    # Add default fields
    for col_name, value in meta.default_fields.items():
        inner = inner.with_columns(pl.lit(value).alias(col_name))

    return inner


def get_parquet_to_darwin_core_column_mapping() -> dict[str, str]:
    """
    Get the mapping from lowercase column names (used in parquet snapshots)
    to camelCase column names (Darwin Core standard).

    Returns:
        Dictionary mapping lowercase column names to camelCase column names
    """
    return _LOWER_TO_CAMEL


def build_taxon_filter(taxon_name: str) -> pl.Expr:
    """
    Build a Polars expression to filter observations by taxon name.

    Checks if the taxon name matches any taxonomic rank column:
    phylum, class, order, family, genus, or species.

    Args:
        taxon_name: The taxon name to filter by

    Returns:
        A Polars expression that matches observations where any taxonomic
        rank column equals the given taxon name
    """
    raise NotImplementedError("Not currently implemented")
    # return (
    #     (pl.col("phylum") == taxon_name)
    #     | (pl.col("class") == taxon_name)
    #     | (pl.col("order") == taxon_name)
    #     | (pl.col("family") == taxon_name)
    #     | (pl.col("genus") == taxon_name)
    #     | (pl.col("species") == taxon_name)
    # )


def build_darwin_core_raw_lf(
    source_path: str,
) -> pl.LazyFrame:
    """
    Build a raw Darwin Core LazyFrame from either a Darwin Core archive or Parquet file.

    This returns unvalidated, unfiltered data before schema validation is applied.
    Automatically detects the source type based on the path.

    Args:
        source_path: Path to either a Darwin Core archive directory or Parquet file/directory

    Returns:
        A LazyFrame with raw Darwin Core data
    """
    # Detect if source is a Darwin Core archive (directory with meta.xml) or parquet
    path = Path(source_path)
    is_darwin_core_archive = path.is_dir() and (path / "meta.xml").exists()

    if is_darwin_core_archive:
        # Load from Darwin Core archive (already uses camelCase)
        inner_lf = scan_darwin_core_archive(source_path)
    else:
        # Load from parquet snapshot and rename columns to camelCase
        # Public GCS buckets (like GBIF) are accessible without credentials
        inner_lf = (
            pl.scan_parquet(
                source_path,
                low_memory=True,
                cache=False,
                parallel="prefiltered",
            )
            .rename(
                get_parquet_to_darwin_core_column_mapping(),
                strict=False,
            )
            # Cast taxonKey from String to UInt32 (GBIF parquet stores it as String)
            .with_columns(pl.col("taxonKey").cast(pl.UInt32))
        )

    # Apply categorical casting (handles both lowercase and camelCase columns)
    inner_lf = cast_taxonomic_columns(inner_lf)

    return inner_lf
