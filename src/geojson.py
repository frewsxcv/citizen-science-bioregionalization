import dataframely as dy
import shapely

import geojson
from src import output
from src.dataframes.cluster_boundary import ClusterBoundarySchema
from src.dataframes.cluster_color import ClusterColorSchema
from src.types import ClusterId, Geocode


def build_geojson_feature(
    geometry: shapely.Geometry,
    cluster: ClusterId,
    color: str,
    darkened_color: str,
) -> geojson.Feature:
    return geojson.Feature(
        properties={
            "color": darkened_color,
            "fillColor": color,
            "fillOpacity": 0.7,
            "weight": 1,
            "cluster": cluster,
        },
        geometry=shapely.geometry.mapping(geometry),  # type: ignore
    )


def build_geojson_feature_collection(
    cluster_boundary_df: dy.DataFrame[ClusterBoundarySchema],
    cluster_colors_df: dy.DataFrame[ClusterColorSchema],
) -> geojson.FeatureCollection:
    features: list[geojson.Feature] = []

    for cluster, boundary, color, darkened_color in cluster_boundary_df.join(
        cluster_colors_df, on="cluster"
    ).iter_rows():
        features.append(
            build_geojson_feature(
                shapely.from_wkb(boundary),
                cluster,
                color,
                darkened_color,
            )
        )
    return geojson.FeatureCollection(features)


def write_geojson(
    feature_collection: geojson.FeatureCollection, output_file: str
) -> None:
    # Prepare the output file path
    output_file = output.prepare_file_path(output_file)

    with open(output_file, "w") as geojson_writer:
        geojson.dump(feature_collection, geojson_writer)
