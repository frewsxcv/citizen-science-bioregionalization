# TODO: Don't include geohashes that extend beyond the bounds of the dataset
# so those clusters will have artificially fewer counts

import logging
import geojson
import typer
from src import cli_output, cluster
from src.cluster_stats import Stats
from src.dataframes.geohash_taxa_counts import GeohashTaxaCountsDataFrame
from src.dataframes.cluster_color import ClusterColorDataFrame
from src.dataframes.taxa_geographic_mean import TaxaGeographicMeanDataFrame
from src.lazyframes.darwin_core_csv import DarwinCoreCsvLazyFrame
from src.render import plot_clusters
from src.geojson import build_geojson_feature_collection


def write_geojson(
    feature_collection: geojson.FeatureCollection, output_file: str
) -> None:
    with open(output_file, "w") as geojson_writer:
        geojson.dump(feature_collection, geojson_writer)


def run(
    geohash_precision: int = typer.Option(..., help="Precision of the geohash"),
    num_clusters: int = typer.Option(..., help="Number of clusters to generate"),
    log_file: str = typer.Option(..., help="Path to the log file"),
    input_file: str = typer.Argument(..., help="Path to the input file"),
    output_file: str = typer.Argument(..., help="Path to the output file"),
    show_dendrogram: bool = typer.Option(False, help="Show the dendrogram"),
    plot: bool = typer.Option(False, help="Plot the clusters"),
    use_cache: bool = typer.Option(False, help="Use the cache"),
):
    logging.basicConfig(filename=log_file, encoding="utf-8", level=logging.INFO)

    darwin_core_csv_lazy_frame = DarwinCoreCsvLazyFrame.from_file(input_file)

    geohash_taxa_counts_dataframe = GeohashTaxaCountsDataFrame.build(
        darwin_core_csv_lazy_frame, geohash_precision
    )

    geohash_cluster_dataframe = cluster.run(
        geohash_taxa_counts_dataframe,
        num_clusters,
        show_dendrogram,
        use_cache,
    )

    taxa_geographic_mean_dataframe = TaxaGeographicMeanDataFrame.build(
        geohash_taxa_counts_dataframe
    )

    # Find the top averages of taxon
    all_stats = Stats.build(geohash_taxa_counts_dataframe)

    cluster_colors_dataframe = ClusterColorDataFrame.from_clusters(
        geohash_cluster_dataframe.cluster_ids()
    )

    feature_collection = build_geojson_feature_collection(
        geohash_cluster_dataframe,
        cluster_colors_dataframe,
    )

    cli_output.print_results(
        geohash_taxa_counts_dataframe, all_stats, geohash_cluster_dataframe
    )

    if plot:
        plot_clusters(feature_collection)

    write_geojson(feature_collection, output_file)


if __name__ == "__main__":
    typer.run(run)
