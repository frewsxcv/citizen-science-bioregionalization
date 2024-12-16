import geohashr
import polars as pl
from src.bbox import Bbox
from src.point import Point

type Geohash = str


def geohash_to_bbox(geohash: str) -> Bbox:
    lat, lon, lat_err, lon_err = geohashr.decode_exact(geohash)
    return Bbox(
        sw=Point(lat=lat - lat_err, lon=lon - lon_err),
        ne=Point(lat=lat + lat_err, lon=lon + lon_err),
    )


def build_geohash_series(
    dataframe: pl.DataFrame, lat_col: pl.Expr, lon_col: pl.Expr, precision: int
) -> pl.DataFrame:
    return dataframe.with_columns(
        pl.struct([lat_col.alias("lat"), lon_col.alias("lon")])
        .map_elements(
            lambda series: geohashr.encode(series["lat"], series["lon"], precision),
            return_dtype=pl.String,
        )
        .alias("geohash")
    )
