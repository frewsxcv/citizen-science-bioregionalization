
import polars as pl
import bioregion_rs
import geojson
from src import output


def build_geojson_feature_collection(
    cluster_boundary_df: pl.DataFrame,
    cluster_colors_df: pl.DataFrame,
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
