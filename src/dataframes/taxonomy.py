import polars as pl
from typing import Self
from src.data_container import DataContainer, assert_dataframe_schema
from polars_darwin_core.darwin_core import kingdom_data_type
from polars_darwin_core import DarwinCoreLazyFrame


class TaxonomyDataFrame(DataContainer):
    """
    A dataframe of taxonomy information. Note that this may include taxa for geocodes that were filtered out.
    """

    df: pl.DataFrame

    SCHEMA = {
        "taxonId": pl.UInt32(),  # Unique identifier for each taxon
        "kingdom": kingdom_data_type,
        "phylum": pl.Categorical(),
        "class": pl.Categorical(),
        "order": pl.Categorical(),
        "family": pl.Categorical(),
        "genus": pl.Categorical(),
        "species": pl.String(),
        "taxonRank": pl.Categorical(),
        "scientificName": pl.String(),
    }

    def __init__(self, df: pl.DataFrame) -> None:
        assert_dataframe_schema(df, self.SCHEMA)
        self.df = df

    @classmethod
    def build(
        cls, darwin_core_csv_lazy_frame: DarwinCoreLazyFrame
    ) -> Self:
        df = (
            darwin_core_csv_lazy_frame._inner.select(
                "kingdom",
                "phylum",
                "class",
                "order",
                "family",
                "genus",
                "species",
                "taxonRank",
                "scientificName",
            )
            .unique()
            .collect()
        )

        # Add a unique taxonId for each row
        df = df.with_row_index("taxonId").cast({
            "taxonId": pl.UInt32(),
            "phylum": pl.Categorical(),
            "class": pl.Categorical(),
            "order": pl.Categorical(),
            "family": pl.Categorical(),
            "genus": pl.Categorical(),
            "species": pl.String(),
            "taxonRank": pl.Categorical(),
        })

        return cls(df)
