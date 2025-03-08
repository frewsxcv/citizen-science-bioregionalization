import geojson
import polars as pl
from typing import List
from typing import Iterator, Tuple
import polars_h3
import shapely
from src.types import Geocode, ClusterId
from src.dataframes.cluster_color import ClusterColorDataFrame
from src.dataframes.geocode_cluster import GeocodeClusterDataFrame


def build_geojson_feature(
    geometry: shapely.Geometry,
    cluster: ClusterId,
    color: str,
) -> geojson.Feature:
    return geojson.Feature(
        properties={
            # "label": ", ".join(geocodes),
            "fill": color,
            "stroke-width": 0,
            "cluster": cluster,
        },
        geometry=shapely.geometry.mapping(geometry),  # type: ignore
    )


def latlng_list_to_lnglat_list(
    latlng_list: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    return [(lng, lat) for lat, lng in latlng_list]


def build_geojson_feature_collection(
    geocode_cluster_dataframe: GeocodeClusterDataFrame,
    cluster_colors_dataframe: ClusterColorDataFrame,
) -> geojson.FeatureCollection:
    features: List[geojson.Feature] = []
    for boundaries, cluster, color in (
        geocode_cluster_dataframe.df.with_columns(
            boundary=polars_h3.cell_to_boundary("geocode")
        )
        .group_by("cluster")
        .agg(pl.col("boundary"))
        .join(cluster_colors_dataframe.df, left_on="cluster", right_on="cluster")
        .select("boundary", "cluster", "color")
        .iter_rows()
    ):
        features.append(
            build_geojson_feature(
                shapely.union_all(
                    [
                        shapely.Polygon(latlng_list_to_lnglat_list(boundary))
                        for boundary in boundaries
                    ]
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
