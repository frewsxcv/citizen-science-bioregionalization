import polars as pl
from typing import Self
from src.data_container import DataContainer

from src.lazyframes.darwin_core_csv import DarwinCoreCsvLazyFrame


class TaxonomyDataFrame(DataContainer):
    df: pl.DataFrame

    SCHEMA = {
        "kingdom": pl.String(),
        "phylum": pl.String(),
        "class": pl.String(),
        "order": pl.String(),
        "family": pl.String(),
        "genus": pl.String(),
        "species": pl.String(),
    }

    def __init__(self, df: pl.DataFrame) -> None:
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
