import dataframely as dy
import polars as pl

from src.constants import KINGDOM_VALUES
from src.dataframes.darwin_core import DarwinCoreSchema
from src.dataframes.geocode import GeocodeNoEdgesSchema
from src.geocode import filter_by_bounding_box, with_geocode_lf
from src.types import Bbox


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
    def build_lf(
        cls,
        darwin_core_csv_lf: dy.LazyFrame["DarwinCoreSchema"],
        geocode_precision: int,
        geocode_lf: dy.LazyFrame[GeocodeNoEdgesSchema],
        bounding_box: Bbox,
    ) -> dy.LazyFrame["TaxonomySchema"]:
        geocodes = (
            geocode_lf.select("geocode")
            .collect(engine="streaming")
            .to_series()
            .to_list()
        )

        lf = (
            darwin_core_csv_lf.pipe(filter_by_bounding_box, bounding_box=bounding_box)
            .pipe(with_geocode_lf, geocode_precision=geocode_precision)
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
            # Add a unique taxonId for each row
            .with_row_index("taxonId")
        )

        return cls.validate(lf, eager=False)
