import geojson
import polars as pl
import shapely
from src.dataframes.geocode_boundary import GeocodeBoundaryDataFrame
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


def build_geojson_feature_collection(
    geocode_boundary_dataframe: GeocodeBoundaryDataFrame,
    geocode_cluster_dataframe: GeocodeClusterDataFrame,
    cluster_colors_dataframe: ClusterColorDataFrame,
) -> geojson.FeatureCollection:
    features: list[geojson.Feature] = []

    for cluster, boundaries, color in (
        geocode_boundary_dataframe
        .df
        .join(geocode_cluster_dataframe.df, on="geocode")
        .drop("geocode")
        .group_by("cluster")
        .agg(pl.all())
        .join(cluster_colors_dataframe.df, on="cluster")
        .iter_rows()
    ):
        features.append(
            build_geojson_feature(
                shapely.union_all([shapely.from_wkb(boundary) for boundary in boundaries]),
                cluster,
                color,
            )
        )
    return geojson.FeatureCollection(features)


def write_geojson(
    feature_collection: geojson.FeatureCollection, output_file: str
) -> None:
    with open(output_file, "w") as geojson_writer:
        geojson.dump(feature_collection, geojson_writer)
