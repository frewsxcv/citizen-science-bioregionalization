import dataframely as dy
import polars as pl

from src.constants import KINGDOM_VALUES
from src.dataframes.geocode import GeocodeNoEdgesSchema
from src.geocode import with_geocode_lazy_frame


class TaxonomySchema(dy.Schema):
    """
    A dataframe of taxonomy information. Note that this may include taxa for geocodes that were filtered out.
    """

    taxonId = dy.UInt32(nullable=False)  # Unique identifier for each taxon
    kingdom = dy.Enum(KINGDOM_VALUES, nullable=True)
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
        cls,
        darwin_core_csv_lazy_frame: pl.LazyFrame,
        geocode_precision: int,
        geocode_lazyframe: dy.LazyFrame[GeocodeNoEdgesSchema],
    ) -> dy.DataFrame["TaxonomySchema"]:
        geocodes = (
            geocode_lazyframe.select("geocode").collect(engine="streaming").to_series()
        )

        df = (
            darwin_core_csv_lazy_frame.pipe(
                with_geocode_lazy_frame, geocode_precision=geocode_precision
            )
            .filter(
                # Ensure geocode exists and is not an edge
                pl.col("geocode").is_in(geocodes)
            )
            .select(
                "kingdom",
                "phylum",
                "class",
                "order",
                "family",
                "genus",
                "species",
                "taxonRank",
                "scientificName",
                # pl.col("acceptedTaxonKey").alias("gbifTaxonId"),
                pl.col("taxonKey").alias("gbifTaxonId"),
            )
            .unique(
                subset=[
                    "scientificName",  # Need to confirm this. Will there be different scientific names for the same GBIF taxon ID?
                    "gbifTaxonId",
                ],
            )
            .collect(engine="streaming")
        )

        # Add a unique taxonId for each row
        df = df.with_row_index("taxonId").cast(
            {
                "taxonId": pl.UInt32(),
                "kingdom": pl.Enum(KINGDOM_VALUES),
                "phylum": pl.Categorical(),
                "class": pl.Categorical(),
                "order": pl.Categorical(),
                "family": pl.Categorical(),
                "genus": pl.Categorical(),
                "species": pl.String(),
                "taxonRank": pl.Categorical(),
                "gbifTaxonId": pl.UInt32(),
            }
        )

        return cls.validate(df)
