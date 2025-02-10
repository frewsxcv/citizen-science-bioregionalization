import polars as pl
from src.darwin_core import kingdom_enum


class DarwinCoreCsvLazyFrame:
    lf: pl.LazyFrame

    SCHEMA_OVERRIDES = {
        "decimalLatitude": pl.Float64(),
        "decimalLongitude": pl.Float64(),
        "taxonKey": pl.UInt64(),
        "verbatimScientificName": pl.String(),
        "order": pl.String(),
        "recordedBy": pl.String(),
        "kingdom": kingdom_enum,
    }

    def __init__(self, lf: pl.LazyFrame):
        self.lf = lf

    @classmethod
    def from_file(cls, csv_path: str):
        return cls(
            pl.scan_csv(
                csv_path,
                has_header=True,
                separator="\t",
                quote_char=None,
                schema_overrides=cls.SCHEMA_OVERRIDES,
                infer_schema=False,
                infer_schema_length=None,
            )
        )
