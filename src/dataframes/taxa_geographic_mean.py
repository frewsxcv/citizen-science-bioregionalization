import polars as pl
import dataframely as dy
from src.dataframes.geocode_taxa_counts import GeocodeTaxaCountsSchema
from src.dataframes.geocode import GeocodeSchema
from src.dataframes.taxonomy import TaxonomySchema
import logging

logger = logging.getLogger(__name__)


class TaxaGeographicMeanSchema(dy.Schema):
    kingdom = dy.Categorical(nullable=True)
    taxonRank = dy.Categorical(nullable=True)
    scientificName = dy.String(nullable=True)
    mean_lat = dy.Float64(nullable=False)
    mean_lon = dy.Float64(nullable=False)

    @classmethod
    def build(
        cls,
        geocode_taxa_counts_dataframe: dy.DataFrame[GeocodeTaxaCountsSchema],
        geocode_dataframe: dy.DataFrame[GeocodeSchema],
        taxonomy_dataframe: dy.DataFrame[TaxonomySchema],
    ) -> dy.DataFrame["TaxaGeographicMeanSchema"]:
        # Join geocode_taxa_counts with geocode_dataframe to get lat/lon
        with_lat_lon = geocode_taxa_counts_dataframe.join(
            geocode_dataframe.select("geocode", "lat", "lon"), on="geocode"
        )

        # Join with taxonomy_dataframe to get taxonomic info
        with_taxonomy = with_lat_lon.join(
            taxonomy_dataframe.select(
                "taxonId", "kingdom", "taxonRank", "scientificName"
            ),
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
