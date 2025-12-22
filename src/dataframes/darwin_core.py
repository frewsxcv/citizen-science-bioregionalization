"""Darwin Core occurrence data schema.

This module defines the schema for validated Darwin Core occurrence data,
ensuring that data loaded from Darwin Core archives or Parquet files
meets the requirements for downstream analysis.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import dataframely as dy
import polars as pl

from src.constants import KINGDOM_DATA_TYPE, KINGDOM_VALUES
from src.types import Bbox


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


# Schema overrides for Darwin Core CSV files
# Polars can infer most types, but some need to be specified
SCHEMA_OVERRIDES: dict[str, type[pl.DataType] | pl.DataType] = {
    "kingdom": KINGDOM_DATA_TYPE,
    "catalogNumber": pl.Utf8,
    "hasCoordinate": pl.Boolean,
    "hasGeospatialIssues": pl.Boolean,
    "repatriated": pl.Boolean,
}


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
        col: SCHEMA_OVERRIDES[col] for col in meta.columns if col in SCHEMA_OVERRIDES
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
        encoding="utf8",
        **scan_csv_kwargs,
    )

    # Add default fields
    for col_name, value in meta.default_fields.items():
        inner = inner.with_columns(pl.lit(value).alias(col_name))

    return inner


def scan_darwin_core_csv(path: Union[str, Path]) -> pl.LazyFrame:
    """Scan a Darwin Core CSV file lazily.

    Args:
        path: Path to the CSV file.

    Returns:
        A Polars LazyFrame containing the Darwin Core data.
    """
    return pl.scan_csv(
        path,
        schema_overrides=SCHEMA_OVERRIDES,
        quote_char=None,
        separator="\t",
    )


def scan_darwin_core_parquet(path: Union[str, Path]) -> pl.LazyFrame:
    """Scan a Darwin Core Parquet file lazily.

    Args:
        path: Path to the Parquet file.

    Returns:
        A Polars LazyFrame containing the Darwin Core data.
    """
    return pl.scan_parquet(path)


class DarwinCoreSchema(dy.Schema):
    """Schema for Darwin Core occurrence records.

    This schema defines the core fields needed for bioregionalization analysis
    from Darwin Core occurrence data. It validates that required geographic
    and taxonomic fields are present and properly typed.
    """

    # Geographic fields (required for spatial analysis)
    decimalLatitude = dy.Float64(nullable=False)
    decimalLongitude = dy.Float64(nullable=False)

    # Taxonomic hierarchy
    kingdom = dy.Enum(KINGDOM_VALUES, nullable=True)
    phylum = dy.Categorical(nullable=True)
    class_ = dy.Categorical(nullable=True, alias="class")
    order = dy.Categorical(nullable=True)
    family = dy.Categorical(nullable=True)
    genus = dy.Categorical(nullable=True)
    species = dy.String(nullable=True)

    # Taxonomic metadata
    taxonRank = dy.Categorical(nullable=True)
    scientificName = dy.String(nullable=True)
    taxonKey = dy.UInt32(nullable=True)

    @dy.rule()
    def valid_latitude(cls) -> pl.Expr:
        """Validate that latitude is within valid range [-90, 90]."""
        return (pl.col("decimalLatitude") >= -90) & (pl.col("decimalLatitude") <= 90)

    @dy.rule()
    def valid_longitude(cls) -> pl.Expr:
        """Validate that longitude is within valid range [-180, 180]."""
        return (pl.col("decimalLongitude") >= -180) & (
            pl.col("decimalLongitude") <= 180
        )

    @classmethod
    def build_lf(
        cls,
        darwin_core_lf: pl.LazyFrame,
        bounding_box: Bbox,
        limit: Union[int, None] = None,
    ) -> dy.LazyFrame["DarwinCoreSchema"]:
        """Build a validated Darwin Core dataframe from a lazy frame.

        Args:
            darwin_core_lf: A LazyFrame containing occurrence data.
            bounding_box: Geographic bounding box to filter records.
            limit: Optional maximum number of records to return.

        Returns:
            A validated DataFrame conforming to DarwinCoreSchema.
        """
        lf = darwin_core_lf

        # Apply geographic bounding box filter
        lf = lf.filter(
            pl.col("decimalLatitude").is_not_null()
            & pl.col("decimalLongitude").is_not_null()
            & (pl.col("decimalLatitude") >= bounding_box.min_lat)
            & (pl.col("decimalLatitude") <= bounding_box.max_lat)
            & (pl.col("decimalLongitude") >= bounding_box.min_lng)
            & (pl.col("decimalLongitude") <= bounding_box.max_lng)
        )

        # Select only the columns we need
        lf = lf.select(
            "decimalLatitude",
            "decimalLongitude",
            "kingdom",
            "phylum",
            "class",
            "order",
            "family",
            "genus",
            "species",
            "taxonRank",
            "scientificName",
            "taxonKey",
        )

        # Apply limit if specified
        if limit is not None:
            lf = lf.limit(limit)

        lf = lf.cast(
            {
                "decimalLatitude": pl.Float64(),
                "decimalLongitude": pl.Float64(),
                "kingdom": pl.Enum(KINGDOM_VALUES),
                "phylum": pl.Categorical(),
                "class": pl.Categorical(),
                "order": pl.Categorical(),
                "family": pl.Categorical(),
                "genus": pl.Categorical(),
                "species": pl.String(),
                "taxonRank": pl.Categorical(),
                "scientificName": pl.String(),
                "taxonKey": pl.UInt32(),
            }
        )

        return cls.validate(lf, eager=False)

    @classmethod
    def from_archive(
        cls,
        path: Union[str, Path],
        bounding_box: Bbox,
        limit: Union[int, None] = None,
    ) -> dy.LazyFrame["DarwinCoreSchema"]:
        """Build a validated Darwin Core lazyframe from an archive directory.

        Args:
            path: Path to an unpacked Darwin Core archive directory
                containing meta.xml and the core data file.
            bounding_box: Geographic bounding box to filter records.
            limit: Optional maximum number of records to return.

        Returns:
            A validated LazyFrame conforming to DarwinCoreSchema.
        """
        darwin_core_lf = scan_darwin_core_archive(path)
        return cls.build_lf(darwin_core_lf, bounding_box, limit)

    @classmethod
    def from_csv(
        cls,
        path: Union[str, Path],
        bounding_box: Bbox,
        limit: Union[int, None] = None,
    ) -> dy.LazyFrame["DarwinCoreSchema"]:
        """Build a validated Darwin Core lazyframe from a CSV file.

        Args:
            path: Path to a Darwin Core CSV file.
            bounding_box: Geographic bounding box to filter records.
            limit: Optional maximum number of records to return.

        Returns:
            A validated LazyFrame conforming to DarwinCoreSchema.
        """
        darwin_core_lf = scan_darwin_core_csv(path)
        return cls.build_lf(darwin_core_lf, bounding_box, limit)

    @classmethod
    def from_parquet(
        cls,
        path: Union[str, Path],
        bounding_box: Bbox,
        limit: Union[int, None] = None,
    ) -> dy.LazyFrame["DarwinCoreSchema"]:
        """Build a validated Darwin Core lazyframe from a Parquet file.

        Args:
            path: Path to a Darwin Core Parquet file.
            bounding_box: Geographic bounding box to filter records.
            limit: Optional maximum number of records to return.

        Returns:
            A validated LazyFrame conforming to DarwinCoreSchema.
        """
        darwin_core_lf = scan_darwin_core_parquet(path)
        return cls.build_lf(darwin_core_lf, bounding_box, limit)
