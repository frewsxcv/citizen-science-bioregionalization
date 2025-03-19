from typing import Any, Dict
from abc import ABC, abstractmethod
import polars as pl


def assert_dataframe_schema(
    df: pl.DataFrame,
    expected_schema: Dict[str, pl.DataType],
) -> None:
    """
    Validates that a DataFrame has the expected schema.

    Args:
        df: The DataFrame to validate
        expected_schema: The expected schema as a dictionary of column names to data types

    Raises:
        AssertionError: If the DataFrame schema doesn't match the expected schema
    """
    assert (
        df.schema == expected_schema
    ), f"Dataframe schema mismatch. Expected: {expected_schema}, got: {df.schema}"


class DataContainer(ABC):
    SCHEMA: Dict[str, pl.DataType]

    @classmethod
    @abstractmethod
    def build(cls, *args: Any, **kwargs: Any) -> "DataContainer":
        pass
