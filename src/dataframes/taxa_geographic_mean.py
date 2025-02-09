import polars as pl
from src.darwin_core import kingdom_enum, TaxonRank
from src.dataframes.geohash_taxa_counts import GeohashTaxaCountsDataFrame
from src.geohash import geohash_to_lat_lon
from typing import Self
import logging

logger = logging.getLogger(__name__)

class TaxaGeographicMeanDataFrame:
    df: pl.DataFrame

    SCHEMA = {
        "kingdom": kingdom_enum,
        "rank": pl.Enum(TaxonRank),
        "name": pl.String(),
        "mean_latitude": pl.Float64(),
        "mean_longitude": pl.Float64(),
    }

    def __init__(self, df: pl.DataFrame):
        self.df = df

    @classmethod
    def build(cls, geohash_taxa_counts: GeohashTaxaCountsDataFrame) -> Self:
        df = geohash_taxa_counts.df.pipe(geohash_to_lat_lon, pl.col("geohash"))

        unique = df["kingdom", "rank", "name"].unique()

        kingdoms = []
        ranks = []
        names = []
        mean_latitudes = []
        mean_longitudes = []

        logger.info("Building taxa geographic mean dataframe")
        for i, row in enumerate(unique.iter_rows(named=True)):
            logger.info(f"{i / len(unique) * 100}%")
            kingdom = row["kingdom"]
            rank = row["rank"]
            name = row["name"]
            filtered_df = df.filter(
                pl.col("kingdom") == kingdom,
                pl.col("rank") == rank,
                pl.col("name") == name,
            )
            # import pdb
            # pdb.set_trace()
            total_count = filtered_df["count"].sum()
            x = 0
            y = 0
            for row in filtered_df.iter_rows(named=True):
                [lat, lon] = row["lat_lon"]
                count = row["count"]
                x += count * lat
                y += count * lon
            centroid = (x / total_count, y / total_count)
            # import pdb
            # pdb.set_trace()
            # Create a single row DataFrame with the results
            kingdoms.append(kingdom)
            ranks.append(rank)
            names.append(name)
            mean_latitudes.append(centroid[0])
            mean_longitudes.append(centroid[1])

        final_df = pl.DataFrame(
            {
                "kingdom": kingdoms,
                "rank": ranks,
                "name": names,
                "mean_latitude": mean_latitudes,
                "mean_longitude": mean_longitudes,
            },
            schema=cls.SCHEMA,
        )
        return cls(final_df)

