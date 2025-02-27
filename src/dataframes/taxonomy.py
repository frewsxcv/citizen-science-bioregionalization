import polars as pl
from typing import Self
from src.data_container import DataContainer
from src.darwin_core import kingdom_enum

from src.lazyframes.darwin_core_csv import DarwinCoreCsvLazyFrame


class TaxonomyDataFrame(DataContainer):
    """
    A dataframe of taxonomy information. Note that this may include taxa for geohashes that were filtered out.
    """

    df: pl.DataFrame

    SCHEMA = {
        "kingdom": kingdom_enum,
        "phylum": pl.String(),
        "class": pl.String(),
        "order": pl.String(),
        "family": pl.String(),
        "genus": pl.String(),
        "species": pl.String(),
    }

    def __init__(self, df: pl.DataFrame) -> None:
        assert df.schema == self.SCHEMA, f"Schema mismatch: {df.schema} != {self.SCHEMA}"
        self.df = df

    @classmethod
    def build(cls, darwin_core_csv_lazy_frame: DarwinCoreCsvLazyFrame) -> Self:
        df = (
            darwin_core_csv_lazy_frame.lf.select(
                "kingdom",
                "phylum",
                "class",
                "order",
                "family",
                "genus",
                "species",
            )
            .unique()
            .collect()
        )
        return cls(df)
