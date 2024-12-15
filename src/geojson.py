import geojson
from typing import List
from src.geohash import geohash_to_bbox, Geohash
import random
from typing import Iterator, Tuple

from src.cluster import ClusterId

COLORS = [
    "#" + "".join([random.choice("0123456789ABCDEF") for _ in range(6)])
    for _ in range(1000)
]


def build_geojson_geohash_polygon(geohash: Geohash) -> geojson.Polygon:
    bbox = geohash_to_bbox(geohash)
    return geojson.Polygon(
        coordinates=[
            [
                [bbox.sw.lon, bbox.sw.lat],
                [bbox.ne.lon, bbox.sw.lat],
                [bbox.ne.lon, bbox.ne.lat],
                [bbox.sw.lon, bbox.ne.lat],
                [bbox.sw.lon, bbox.sw.lat],
            ]
        ]
    )


def build_geojson_feature(
    geohashes: List[Geohash], cluster: ClusterId
) -> geojson.Feature:
    geometries = [build_geojson_geohash_polygon(geohash) for geohash in geohashes]
    geometry = (
        geojson.GeometryCollection(geometries) if len(geometries) > 1 else geometries[0]
    )

    return geojson.Feature(
        properties={
            "label": ", ".join(geohashes),
            "fill": COLORS[cluster],
            "stroke-width": 0,
            "cluster": cluster,
        },
        geometry=geometry,
    )


def build_geojson_feature_collection(
    cluster_and_geohashes: Iterator[Tuple[ClusterId, List[Geohash]]],
) -> geojson.FeatureCollection:
    return geojson.FeatureCollection(
        features=[
            build_geojson_feature(geohashes, cluster)
            for cluster, geohashes in cluster_and_geohashes
        ],
    )
