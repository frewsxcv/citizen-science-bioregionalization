import geojson
import polars as pl
from typing import List
from src.geohash import geohash_to_bbox, Geohash
from typing import Iterator, Tuple
import shapely
from src.types import Geohash, ClusterId
from src.dataframes.cluster_color import ClusterColorDataFrame
from src.dataframes.geohash_cluster import GeohashClusterDataFrame


def build_geojson_geohash_polygon(geohash: Geohash) -> shapely.Polygon:
    bbox = geohash_to_bbox(geohash)
    return shapely.Polygon(
        [
            (bbox.sw.x, bbox.sw.y),
            (bbox.ne.x, bbox.sw.y),
            (bbox.ne.x, bbox.ne.y),
            (bbox.sw.x, bbox.ne.y),
            (bbox.sw.x, bbox.sw.y),
        ]
    )


def build_geojson_feature(
    geometry: shapely.Geometry,
    cluster: ClusterId,
    color: str,
) -> geojson.Feature:
    return geojson.Feature(
        properties={
            # "label": ", ".join(geohashes),
            "fill": color,
            "stroke-width": 0,
            "cluster": cluster,
        },
        geometry=shapely.geometry.mapping(geometry),  # type: ignore
    )


def build_geojson_feature_collection(
    geohash_cluster_dataframe: GeohashClusterDataFrame,
    cluster_colors_dataframe: ClusterColorDataFrame,
) -> geojson.FeatureCollection:
    features: List[geojson.Feature] = []
    for cluster, geohashes, color in (
        geohash_cluster_dataframe.df.group_by("cluster")
        .agg(pl.col("geohash"))
        .join(cluster_colors_dataframe.df, left_on="cluster", right_on="cluster")
        .iter_rows()
    ):
        features.append(
            build_geojson_feature(
                shapely.union_all(
                    [build_geojson_geohash_polygon(geohash) for geohash in geohashes]
                ),
                cluster,
                color,
            )
        )
    return geojson.FeatureCollection(features=features)


def write_geojson(
    feature_collection: geojson.FeatureCollection, output_file: str
) -> None:
    with open(output_file, "w") as geojson_writer:
        geojson.dump(feature_collection, geojson_writer)
