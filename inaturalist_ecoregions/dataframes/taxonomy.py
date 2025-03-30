import polars as pl
from inaturalist_ecoregions.data_container import DataContainer, assert_dataframe_schema
from inaturalist_ecoregions.darwin_core import kingdom_enum

from inaturalist_ecoregions.lazyframes.darwin_core_csv import DarwinCoreCsvLazyFrame
import rust_dataframe_utils # Import the Rust extension module


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
    def build(
        cls, darwin_core_csv_lazy_frame: DarwinCoreCsvLazyFrame
    ) -> "TaxonomyDataFrame":
        # Call the Rust function to perform the dataframe operations
        df = rust_dataframe_utils.build_taxonomy_dataframe_rust(
            darwin_core_csv_lazy_frame.lf
        )

        # The Rust function returns a Polars DataFrame directly
        return cls(df)
