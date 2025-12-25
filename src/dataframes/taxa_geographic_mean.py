import logging

import dataframely as dy
import polars as pl

from src.constants import KINGDOM_VALUES, TAXON_RANK_VALUES
from src.dataframes.geocode import GeocodeNoEdgesSchema
from src.dataframes.geocode_taxa_counts import GeocodeTaxaCountsSchema
from src.dataframes.taxonomy import TaxonomySchema

logger = logging.getLogger(__name__)


class TaxaGeographicMeanSchema(dy.Schema):
    kingdom = dy.Enum(KINGDOM_VALUES, nullable=True)
    taxonRank = dy.Enum(TAXON_RANK_VALUES, nullable=False)
    scientificName = dy.String(nullable=True)
    mean_lat = dy.Float64(nullable=False)
    mean_lon = dy.Float64(nullable=False)

    @classmethod
    def build_df(
        cls,
        geocode_taxa_counts_df: dy.DataFrame[GeocodeTaxaCountsSchema],
        geocode_df: dy.DataFrame[GeocodeNoEdgesSchema],
        taxonomy_df: dy.DataFrame[TaxonomySchema],
    ) -> dy.DataFrame["TaxaGeographicMeanSchema"]:
        # Join geocode_taxa_counts with geocode_df to get lat/lon
        with_lat_lon = geocode_taxa_counts_df.join(
            geocode_df.select("geocode", "lat", "lon"),
            on="geocode",
        )

        # Join with taxonomy_df to get taxonomic info
        with_taxonomy = with_lat_lon.join(
            taxonomy_df.select("taxonId", "kingdom", "taxonRank", "scientificName"),
            on="taxonId",
        )

        # TODO: this doesn't handle the international date line
        df = (
            with_taxonomy.lazy()
            .with_columns(
                (pl.col("lat") * pl.col("count")).alias("lat_scaled"),
                (pl.col("lon") * pl.col("count")).alias("lon_scaled"),
            )
            .group_by("kingdom", "taxonRank", "scientificName")
            .agg(
                (pl.col("lat_scaled").sum() / pl.col("count").sum()).alias("mean_lat"),
                (pl.col("lon_scaled").sum() / pl.col("count").sum()).alias("mean_lon"),
            )
            .collect()
        )
        return cls.validate(df)
