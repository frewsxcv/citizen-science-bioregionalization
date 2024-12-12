import csv
from typing import Dict, Generator, NamedTuple, Optional, Self
import logging
import polars as pl
from src.point import Point
from contexttimer import Timer

logger = logging.getLogger(__name__)

type TaxonId = int


def read_rows(input_file: str) -> Generator[pl.DataFrame, None, None]:
    with open(input_file, "rb") as f:
        num_lines = sum(1 for _ in f)

    logger.info(f"Processing CSV file ({num_lines} lines)")

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
    batches = reader.next_batches(n=5)
    processed_lines = 0
    while batches:
        for batch in batches:
            with Timer(
                output=logger.info,
                prefix=f"Processing CSV file: {processed_lines}/{num_lines} lines ({(processed_lines/num_lines)*100:.1f}%)",
            ):
                yield batch
            processed_lines += len(batch)
        batches = reader.next_batches(n=5)
