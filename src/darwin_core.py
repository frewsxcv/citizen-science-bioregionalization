import csv
from typing import Dict, Generator, NamedTuple, Optional, Self
import logging
import pandas as pd
from src.point import Point
from contexttimer import Timer

logger = logging.getLogger(__name__)

type TaxonId = int


dtype = {
    "decimalLatitude": "Float64",
    "decimalLongitude": "Float64",
    "taxonKey": "UInt64",
}


def read_rows(input_file: str) -> Generator[pd.DataFrame, None, None]:
    with open(input_file, "rb") as f:
        num_lines = sum(1 for _ in f)

    logger.info(f"Processing CSV file ({num_lines} lines)")

    chunksize = 200_000
    num_chunks = num_lines // chunksize

    for i, chunk in enumerate(pd.read_csv(
        input_file,
        delimiter="\t",
        chunksize=200_000,
        dtype=dtype,
        usecols=[
            "decimalLatitude",
            "decimalLongitude",
            "taxonKey",
            "verbatimScientificName",
            "order",
            "recordedBy",
        ],
    )):
        with Timer(output=logger.info, prefix=f"Processing CSV file chunk {i}/{num_chunks} ({(i/num_chunks)*100:.1f}%)"):
            yield chunk

    # with open(input_file, "r") as f:
    #     reader = csv.DictReader(f, delimiter="\t")
    #     for dict_row in reader:
    #         row = Row.from_csv_dict(dict_row)
    #         if row:
    #             yield row
