import dataframely as dy
import polars as pl
from polars_darwin_core import DarwinCoreLazyFrame, Kingdom


class TaxonomySchema(dy.Schema):
    """
    A dataframe of taxonomy information. Note that this may include taxa for geocodes that were filtered out.
    """

    taxonId = dy.UInt32(nullable=False)  # Unique identifier for each taxon
    kingdom = dy.Enum(Kingdom, nullable=True)
    phylum = dy.Categorical(nullable=True)
    class_ = dy.Categorical(nullable=True, alias="class")
    order = dy.Categorical(nullable=True)
    family = dy.Categorical(nullable=True)
    genus = dy.Categorical(nullable=True)
    species = dy.String(nullable=True)
    taxonRank = dy.Categorical(nullable=True)
    scientificName = dy.String(nullable=True)
    gbifTaxonId = dy.UInt32(nullable=True)

    @classmethod
    def build(
        cls, darwin_core_csv_lazy_frame: DarwinCoreLazyFrame
    ) -> dy.DataFrame["TaxonomySchema"]:
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
                "acceptedTaxonKey",
            )
            .rename({"acceptedTaxonKey": "gbifTaxonId"})
            .unique()
            .collect()
        )

        # Add a unique taxonId for each row
        df = df.with_row_index("taxonId").cast(
            {
                "taxonId": pl.UInt32(),
                "kingdom": pl.Enum(Kingdom),
                "phylum": pl.Categorical(),
                "class": pl.Categorical(),
                "order": pl.Categorical(),
                "family": pl.Categorical(),
                "genus": pl.Categorical(),
                "species": pl.String(),
                "taxonRank": pl.Categorical(),
            }
        )

        return cls.validate(df)
