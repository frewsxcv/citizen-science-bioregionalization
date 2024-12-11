import pygeohash

from src.bbox import Bbox
from src.point import Point

type Geohash = str


def geohash_to_bbox(geohash: str) -> Bbox:
    lat, lon, lat_err, lon_err = pygeohash.decode_exactly(geohash)
    return Bbox(
        sw=Point(lat=lat - lat_err, lon=lon - lon_err),
        ne=Point(lat=lat + lat_err, lon=lon + lon_err),
    )
