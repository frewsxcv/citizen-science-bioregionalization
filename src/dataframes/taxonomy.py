import polars as pl
from src.data_container import DataContainer, assert_dataframe_schema
from src.darwin_core import kingdom_enum

from src.lazyframes.darwin_core_csv import DarwinCoreCsvLazyFrame


class TaxonomyDataFrame(DataContainer):
    """
    A dataframe of taxonomy information. Note that this may include taxa for geocodes that were filtered out.
    """

    df: pl.DataFrame

    SCHEMA = {
        "taxonId": pl.UInt32(),  # Unique identifier for each taxon
        "kingdom": kingdom_enum,
        "phylum": pl.String(),
        "class": pl.String(),
        "order": pl.String(),
        "family": pl.String(),
        "genus": pl.String(),
        "species": pl.String(),
        "taxonRank": pl.String(),
        "scientificName": pl.String(),
    }

    def __init__(self, df: pl.DataFrame) -> None:
        assert_dataframe_schema(df, self.SCHEMA)
        self.df = df

    @classmethod
    def build(cls, darwin_core_csv_lazy_frame: DarwinCoreCsvLazyFrame) -> 'TaxonomyDataFrame':
        df = (
            darwin_core_csv_lazy_frame.lf.select(
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
        df = df.with_row_index("taxonId").with_columns(
            pl.col("taxonId").cast(pl.UInt32())
        )
        
        return cls(df)
