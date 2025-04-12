# TODO: Don't include geocodes that extend beyond the bounds of the dataset
# so those clusters will have artificially fewer counts

import logging
import argparse
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
from src.dataframes.cluster_significant_differences import (
    ClusterSignificantDifferencesDataFrame,
)
from src.dataframes.permanova_results import PermanovaResultsDataFrame
from src.matrices.geocode_distance import GeocodeDistanceMatrix
from src.matrices.geocode_connectivity import GeocodeConnectivityMatrix
from src.lazyframes.darwin_core_csv import DarwinCoreCsvLazyFrame
from src.render import plot_clusters
from src.geojson import build_geojson_feature_collection, write_geojson
from src.dataframes.geocode import GeocodeDataFrame
from src.html_output import prepare_full_report_data, render_html, write_html
import os
from src import output
from typing import Optional


def run(
    geohash_precision: int,
    num_clusters: int,
    log_file: str,
    input_file: str,
    taxon_filter: Optional[str] = None,
):
    # Get standardized output paths
    output_file = output.get_geojson_path()
    html_output = output.get_html_path()

    # Normalize log file path to ensure it's in the output directory
    log_file = output.normalize_path(log_file)

    # Ensure output directory exists
    output.ensure_output_dir()

    logging.basicConfig(filename=log_file, encoding="utf-8", level=logging.INFO)

    darwin_core_csv_lazy_frame = DarwinCoreCsvLazyFrame.build(
        input_file, taxon_filter=taxon_filter
    )

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
        taxonomy_dataframe,
    )

    distance_matrix = GeocodeDistanceMatrix.build(
        geocode_taxa_counts_dataframe,
        geocode_dataframe,
    )

    connectivity_matrix = GeocodeConnectivityMatrix.build(geocode_dataframe)

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

    geocode_boundary_dataframe = GeocodeBoundaryDataFrame.build(
        geocode_cluster_dataframe,
    )

    cluster_boundary_dataframe = ClusterBoundaryDataFrame.build(
        geocode_cluster_dataframe,
        geocode_boundary_dataframe,
    )

    # Use the updated build method to color ocean clusters blue
    cluster_colors_dataframe = ClusterColorDataFrame.build(
        cluster_neighbors_dataframe,
        cluster_boundary_dataframe,
    )

    feature_collection = build_geojson_feature_collection(
        cluster_boundary_dataframe,
        cluster_colors_dataframe,
    )

    # Calculate PERMANOVA results
    permanova_results_dataframe = PermanovaResultsDataFrame.build(
        geocode_distance_matrix=distance_matrix,
        geocode_cluster_dataframe=geocode_cluster_dataframe,
        geocode_dataframe=geocode_dataframe,
    )

    # Print CLI results, including PERMANOVA
    cli_output.print_results(
        geocode_cluster_dataframe=geocode_cluster_dataframe,
        permanova_results_dataframe=permanova_results_dataframe,
    )

    write_geojson(feature_collection, output_file)

    # Calculate significant differences for HTML output
    # ClusterSignificantDifferencesDataFrame only needs the all_stats parameter
    cluster_significant_differences_df = ClusterSignificantDifferencesDataFrame.build(
        all_stats
    )

    # Generate HTML with maps
    report_data = prepare_full_report_data(
        cluster_colors_dataframe,
        cluster_significant_differences_df,
        taxonomy_dataframe,
        feature_collection,
    )
    html_content = render_html("cluster_report.html", report_data)

    # Write HTML to file
    write_html(html_content, html_output)
    logging.info(f"HTML output with maps written to {html_output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process Darwin Core CSV data and generate clusters."
    )

    # Add required options
    parser.add_argument(
        "--geohash-precision", type=int, required=True, help="Precision of the geocode"
    )
    parser.add_argument(
        "--num-clusters", type=int, required=True, help="Number of clusters to generate"
    )
    parser.add_argument(
        "--log-file", type=str, required=True, help="Path to the log file"
    )

    # Add optional arguments
    parser.add_argument(
        "--taxon-filter",
        type=str,
        default=None,
        help="Filter to a specific taxon (e.g., 'Aves')",
    )

    # Positional arguments
    parser.add_argument("input_file", type=str, help="Path to the input file")

    args = parser.parse_args()

    run(
        geohash_precision=args.geohash_precision,
        num_clusters=args.num_clusters,
        log_file=args.log_file,
        input_file=args.input_file,
        taxon_filter=args.taxon_filter,
    )


if __name__ == "__main__":
    main()
