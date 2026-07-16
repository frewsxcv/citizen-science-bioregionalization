import dataframely as dy

import bioregion_rs
import geojson
from src import output
from src.dataframes.cluster_boundary import ClusterBoundarySchema
from src.dataframes.cluster_color import ClusterColorSchema


def build_geojson_feature_collection(
    cluster_boundary_df: dy.DataFrame[ClusterBoundarySchema],
    cluster_colors_df: dy.DataFrame[ClusterColorSchema],
) -> geojson.FeatureCollection:
    rust_json = bioregion_rs.build_geojson_feature_collection(
        cluster_boundary_df, cluster_colors_df
    )
    return geojson.loads(rust_json)


def write_geojson(
    feature_collection: geojson.FeatureCollection, output_file: str
) -> None:
    # Prepare the output file path
    output_file = output.prepare_file_path(output_file)

    with open(output_file, "w") as geojson_writer:
        geojson.dump(feature_collection, geojson_writer)
