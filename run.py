# TODO: Don't include geohashes that extend beyond the bounds of the dataset
# so those clusters will have artificially fewer counts

import logging
import typer
from src import cli_output
from src.dataframes.cluster_taxa_statistics import ClusterTaxaStatisticsDataFrame
from src.dataframes.geohash_cluster import GeohashClusterDataFrame
from src.dataframes.geohash_species_counts import GeohashSpeciesCountsDataFrame
from src.dataframes.cluster_color import ClusterColorDataFrame
from src.dataframes.taxa_geographic_mean import TaxaGeographicMeanDataFrame
from src.dataframes.taxonomy import TaxonomyDataFrame
from src.distance_matrix import DistanceMatrix
from src.lazyframes.darwin_core_csv import DarwinCoreCsvLazyFrame
from src.render import plot_clusters
from src.geojson import build_geojson_feature_collection, write_geojson


def run(
    geohash_precision: int = typer.Option(..., help="Precision of the geohash"),
    num_clusters: int = typer.Option(..., help="Number of clusters to generate"),
    log_file: str = typer.Option(..., help="Path to the log file"),
    input_file: str = typer.Argument(..., help="Path to the input file"),
    output_file: str = typer.Argument(..., help="Path to the output file"),
    plot: bool = typer.Option(False, help="Plot the clusters"),
    use_cache: bool = typer.Option(False, help="Use the cache"),
):
    logging.basicConfig(filename=log_file, encoding="utf-8", level=logging.INFO)

    darwin_core_csv_lazy_frame = DarwinCoreCsvLazyFrame.from_file(input_file)

    taxonomy_dataframe = TaxonomyDataFrame.build(
        darwin_core_csv_lazy_frame,
    )

    geohash_taxa_counts_dataframe = GeohashSpeciesCountsDataFrame.build(
        darwin_core_csv_lazy_frame, geohash_precision
    )

    distance_matrix = DistanceMatrix.build(
        geohash_taxa_counts_dataframe,
        use_cache,
    )

    geohash_cluster_dataframe = GeohashClusterDataFrame.build(
        geohash_taxa_counts_dataframe,
        distance_matrix,
        num_clusters,
    )

    taxa_geographic_mean_dataframe = TaxaGeographicMeanDataFrame.build(
        geohash_taxa_counts_dataframe
    )

    # Find the top averages of taxon
    all_stats = ClusterTaxaStatisticsDataFrame.build(
        geohash_taxa_counts_dataframe,
        geohash_cluster_dataframe,
        taxonomy_dataframe,
    )

    cluster_colors_dataframe = ClusterColorDataFrame.from_clusters(
        geohash_cluster_dataframe
    )

    feature_collection = build_geojson_feature_collection(
        geohash_cluster_dataframe,
        cluster_colors_dataframe,
    )

    cli_output.print_results(all_stats, geohash_cluster_dataframe)

    if plot:
        plot_clusters(feature_collection)

    write_geojson(feature_collection, output_file)


if __name__ == "__main__":
    typer.run(run)
