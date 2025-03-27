import polars as pl
import logging
from typing import Optional, List
from src.darwin_core import kingdom_enum, TAXONOMIC_RANKS
from src.data_container import DataContainer

logger = logging.getLogger(__name__)


class DarwinCoreCsvLazyFrame(DataContainer):
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
    def build(
        cls, csv_path: str, taxon_filter: Optional[str] = None
    ) -> "DarwinCoreCsvLazyFrame":
        lf = pl.scan_csv(
            csv_path,
            has_header=True,
            separator="\t",
            quote_char=None,
            schema_overrides=cls.SCHEMA_OVERRIDES,
            infer_schema=False,
            infer_schema_length=None,
        )

        # Apply taxon filter if provided
        if taxon_filter:
            logger.info(f"Filtering data to taxon: {taxon_filter}")

            # Convert taxon_filter to lowercase for case-insensitive comparison
            taxon_filter_lower = taxon_filter.lower()

            # Create a filter for any rank matching the taxon (case insensitive)
            conditions = []
            for rank in TAXONOMIC_RANKS:
                # Special handling for enum types (like kingdom)
                if rank == "kingdom":
                    # Cast enum to string before applying string operations
                    condition = (
                        pl.col(rank)
                        .cast(pl.Utf8)
                        .str.to_lowercase()
                        .eq(taxon_filter_lower)
                    )
                else:
                    # Regular string columns
                    condition = pl.col(rank).str.to_lowercase().eq(taxon_filter_lower)
                conditions.append(condition)

            # Combine all conditions with OR
            combined_filter = conditions[0]
            for condition in conditions[1:]:
                combined_filter = combined_filter | condition

            # Apply the filter
            lf = lf.filter(combined_filter)

        return cls(lf)
