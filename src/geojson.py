import geojson
import polars as pl
from typing import List
from src.geocode import geohash_to_bbox, Geocode
from typing import Iterator, Tuple
import shapely
from src.types import Geocode, ClusterId
from src.dataframes.cluster_color import ClusterColorDataFrame
from src.dataframes.geocode_cluster import GeocodeClusterDataFrame


def build_geojson_geocode_polygon(geocode: Geocode) -> shapely.Polygon:
    bbox = geohash_to_bbox(geocode)
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
            # "label": ", ".join(geocodees),
            "fill": color,
            "stroke-width": 0,
            "cluster": cluster,
        },
        geometry=shapely.geometry.mapping(geometry),  # type: ignore
    )


def build_geojson_feature_collection(
    geocode_cluster_dataframe: GeocodeClusterDataFrame,
    cluster_colors_dataframe: ClusterColorDataFrame,
) -> geojson.FeatureCollection:
    features: List[geojson.Feature] = []
    for cluster, geocodees, color in (
        geocode_cluster_dataframe.df.group_by("cluster")
        .agg(pl.col("geocode"))
        .join(cluster_colors_dataframe.df, left_on="cluster", right_on="cluster")
        .iter_rows()
    ):
        features.append(
            build_geojson_feature(
                shapely.union_all(
                    [build_geojson_geocode_polygon(geocode) for geocode in geocodees]
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
