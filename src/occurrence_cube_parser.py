import polars as pl
from typing import Optional

def parse_occurrence_cube(file_path: str) -> pl.DataFrame:
    """
    Parses a b-cubed species occurrence cube file (tab-separated CSV) using polars.

    Args:
        file_path: The path to the occurrence cube file.

    Returns:
        A polars DataFrame containing the parsed data.
    """
    # The file is tab-separated, so we specify the separator
    # Define expected column names and their data types
    expected_schema = {
        "kingdom": pl.Utf8,
        "kingdomkey": pl.Int64,
        "phylum": pl.Utf8,
        "phylumkey": pl.Int64,
        "class": pl.Utf8,
        "classkey": pl.Int64,
        "order": pl.Utf8,
        "orderkey": pl.Int64,
        "family": pl.Utf8,
        "familykey": pl.Int64,
        "genus": pl.Utf8,
        "genuskey": pl.Int64,
        "species": pl.Utf8,
        "specieskey": pl.Int64,
        "year": pl.Int64,
        "isea3hcellcode": pl.Int64,
        "kingdomcount": pl.Int64,
        "phylumcount": pl.Int64,
        "classcount": pl.Int64,
        "ordercount": pl.Int64,
        "familycount": pl.Int64,
        "genuscount": pl.Int64,
        "occurrences": pl.Int64,
        "mintemporaluncertainty": pl.Int64,
        "mincoordinateuncertaintyinmeters": pl.Float64
    }

    # The file is tab-separated, so we specify the separator and the schema
    df = pl.read_csv(file_path, separator='\t', schema=expected_schema)

    # Assert that the DataFrame schema matches the expected schema
    assert df.schema == expected_schema, f"Unexpected schema in the DataFrame. Expected: {expected_schema}, Got: {df.schema}"

    return df
