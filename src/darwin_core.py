from typing import Generator
import logging
import polars as pl
from contexttimer import Timer
import os

logger = logging.getLogger(__name__)

type TaxonId = int


def read_rows(
    input_file: str, n_batches: int = 5
) -> Generator[pl.DataFrame, None, None]:
    # Get file size
    file_size = os.path.getsize(input_file)

    # Skipping the header, read first two five lines to estimate average line length
    with open(input_file, "rb") as f:
        # Skip header
        f.readline()
        avg_line_size = sum(len(f.readline()) for _ in range(5)) / 5

    # Estimate total lines (subtract 1 for header)
    num_lines = (file_size // avg_line_size) - 1

    logger.info(f"Processing CSV file (estimated {num_lines:,} lines)")

    reader = pl.read_csv_batched(
        input_file,
        separator="\t",
        schema_overrides={
            "decimalLatitude": pl.Float64,
            "decimalLongitude": pl.Float64,
            "taxonKey": pl.UInt64,
            "verbatimScientificName": pl.String,
            "order": pl.String,
            "recordedBy": pl.String,
        },
        infer_schema_length=0,
        quote_char=None,
        columns=[
            "decimalLatitude",
            "decimalLongitude",
            "taxonKey",
            "verbatimScientificName",
            "order",
            "recordedBy",
        ],
    )
    batches = reader.next_batches(n=n_batches)
    processed_lines = 0
    last_percentage = -1  # Track the last logged percentage
    while batches:
        for batch in batches:
            current_percentage = int((processed_lines / num_lines) * 100)
            processed_lines += len(batch)
            # Only log if percentage has increased by at least 1%
            if current_percentage > last_percentage:
                with Timer(
                    output=logger.info,
                    prefix=f"Processed CSV file: {processed_lines} lines (~{current_percentage}%)",
                ):
                    yield batch
                last_percentage = current_percentage
            else:
                yield batch
        batches = reader.next_batches(n=n_batches)
