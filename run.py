# TODO: Don't include geocodes that extend beyond the bounds of the dataset
# so those clusters will have artificially fewer counts

import logging
import typer
from src import cli_output
from src.dataframes.cluster_neighbors import ClusterNeighborsDataFrame
from src.dataframes.cluster_taxa_statistics import ClusterTaxaStatisticsDataFrame
from src.dataframes.geocode_boundary import GeocodeBoundaryDataFrame
from src.dataframes.geocode_cluster import GeocodeClusterDataFrame
from src.dataframes.geocode_taxa_counts import GeocodeTaxaCountsDataFrame
from src.dataframes.cluster_color import ClusterColorDataFrame
from src.dataframes.cluster_boundary import ClusterBoundaryDataFrame
from src.dataframes.taxa_geographic_mean import TaxaGeographicMeanDataFrame
from src.dataframes.taxonomy import TaxonomyDataFrame
from src.matrices.distance import DistanceMatrix
from src.matrices.connectivity import ConnectivityMatrix
from src.lazyframes.darwin_core_csv import DarwinCoreCsvLazyFrame
from src.render import plot_clusters
from src.geojson import build_geojson_feature_collection, write_geojson
from src.dataframes.geocode import GeocodeDataFrame


def run(
    geohash_precision: int = typer.Option(..., help="Precision of the geocode"),
    num_clusters: int = typer.Option(..., help="Number of clusters to generate"),
    log_file: str = typer.Option(..., help="Path to the log file"),
    input_file: str = typer.Argument(..., help="Path to the input file"),
    output_file: str = typer.Argument(..., help="Path to the output file"),
    plot: bool = typer.Option(False, help="Plot the clusters"),
):
    logging.basicConfig(filename=log_file, encoding="utf-8", level=logging.INFO)

    darwin_core_csv_lazy_frame = DarwinCoreCsvLazyFrame.build(input_file)

    geocode_dataframe = GeocodeDataFrame.build(
        darwin_core_csv_lazy_frame,
        geohash_precision,
    )

    taxonomy_dataframe = TaxonomyDataFrame.build(
        darwin_core_csv_lazy_frame,
    )

    geocode_taxa_counts_dataframe = GeocodeTaxaCountsDataFrame.build(
        darwin_core_csv_lazy_frame,
        geohash_precision,
    )

    distance_matrix = DistanceMatrix.build(
        geocode_taxa_counts_dataframe,
        geocode_dataframe,
    )

    connectivity_matrix = ConnectivityMatrix.build(geocode_dataframe)

    geocode_cluster_dataframe = GeocodeClusterDataFrame.build(
        geocode_dataframe,
        distance_matrix,
        connectivity_matrix,
        num_clusters,
    )

    # taxa_geographic_mean_dataframe = TaxaGeographicMeanDataFrame.build(
    #     geocode_taxa_counts_dataframe
    # )

    # Find the top averages of taxon
    all_stats = ClusterTaxaStatisticsDataFrame.build(
        geocode_taxa_counts_dataframe,
        geocode_cluster_dataframe,
        taxonomy_dataframe,
    )

    cluster_neighbors_dataframe = ClusterNeighborsDataFrame.build(
        geocode_dataframe,
        geocode_cluster_dataframe,
    )

    cluster_colors_dataframe = ClusterColorDataFrame.build(
        cluster_neighbors_dataframe,
    )

    geocode_boundary_dataframe = GeocodeBoundaryDataFrame.build(
        geocode_cluster_dataframe,
    )

    cluster_boundary_dataframe = ClusterBoundaryDataFrame.build(
        geocode_cluster_dataframe,
        geocode_boundary_dataframe,
    )

    feature_collection = build_geojson_feature_collection(
        cluster_boundary_dataframe,
        cluster_colors_dataframe,
    )

    cli_output.print_results(all_stats, geocode_cluster_dataframe)

    write_geojson(feature_collection, output_file)

    if plot:
        plot_clusters(feature_collection)


def main() -> None:
    typer.run(run)


if __name__ == "__main__":
    main()
