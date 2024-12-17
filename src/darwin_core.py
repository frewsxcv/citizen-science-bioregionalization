from typing import Generator, List
import logging
import polars as pl
import os


SCHEMA = {
    "decimalLatitude": pl.Float64,
    "decimalLongitude": pl.Float64,
    "taxonKey": pl.UInt64,
    "verbatimScientificName": pl.String,
    "order": pl.String,
    "recordedBy": pl.String,
}


class ProgressLogger:
    logger: logging.Logger
    num_lines: int
    processed_lines: int
    last_percentage: int

    def __init__(self, input_file: str):
        self.logger = logging.getLogger(__name__)
        file_size = os.path.getsize(input_file)
        # Skipping the header, read first two five lines to estimate average line length
        with open(input_file, "rb") as f:
            f.readline()  # Skip header
            avg_line_size = sum(len(f.readline()) for _ in range(5)) / 5
        self.num_lines = int((file_size // avg_line_size) - 1)
        self.processed_lines = 0
        self.last_percentage = -1
        self.logger.info(f"Processing CSV file (estimated {self.num_lines} lines)")

    def log_progress(self):
        current_percentage = int((self.processed_lines / self.num_lines) * 100)
        # Only log if percentage has increased by at least 1%
        if current_percentage > self.last_percentage:
            self.logger.info(
                f"Processed CSV file: {self.processed_lines} lines (~{current_percentage}%)"
            )
            self.last_percentage = current_percentage

    def update(self, lines_processed: int):
        self.processed_lines += lines_processed
        self.log_progress()


class VoidLogger:
    def log_progress(self):
        pass

    def update(self, lines_processed: int):
        pass


def read_rows(
    input_file: str,
    columns: List[str],
    n_batches: int = 5,
    log_progress: bool = True,
) -> Generator[pl.DataFrame, None, None]:
    logger = ProgressLogger(input_file) if log_progress else VoidLogger()

    reader = pl.read_csv_batched(
        input_file,
        separator="\t",
        schema_overrides=SCHEMA,
        infer_schema_length=0,
        quote_char=None,
        columns=columns,
    )
    batches = reader.next_batches(n=n_batches)
    while batches:
        for batch in batches:
            logger.update(len(batch))
            yield batch
        batches = reader.next_batches(n=n_batches)
