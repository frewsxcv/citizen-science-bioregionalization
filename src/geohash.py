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
    dataframe: pl.LazyFrame, lat_col: pl.Expr, lon_col: pl.Expr, precision: int
) -> pl.LazyFrame:
    return dataframe.with_columns(
        pl.struct([lat_col, lon_col])
        .map_elements(
            lambda series: geohashr.encode(series[lat_col], series[lon_col], precision),
            return_dtype=pl.String,
        )
        .alias("geohash")
    )
