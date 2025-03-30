import polars as pl
from inaturalist_ecoregions.dataframes.cluster_color import ClusterColorDataFrame
from inaturalist_ecoregions.dataframes.cluster_significant_differences import (
    ClusterSignificantDifferencesDataFrame,
)
from inaturalist_ecoregions.dataframes.taxonomy import TaxonomyDataFrame
from inaturalist_ecoregions.render import plot_single_cluster, plot_entire_region
import geojson
import os
import jinja2
import json
import base64
from inaturalist_ecoregions import output


def prepare_cluster_data(
    cluster_colors_dataframe: ClusterColorDataFrame,
    significant_differences_df: ClusterSignificantDifferencesDataFrame,
    taxonomy_df: TaxonomyDataFrame,
) -> list:
    """
    Prepare structured data for clusters without any HTML templating.

    Args:
        cluster_colors_dataframe: DataFrame with cluster colors
        significant_differences_df: DataFrame with significant taxonomic differences
        taxonomy_df: DataFrame with taxonomy information

    Returns:
        List of dictionaries with cluster data
    """
    clusters_data = []

    for cluster, color in cluster_colors_dataframe.df.select(
        ["cluster", "color"]
    ).iter_rows(named=False):
        cluster_data = {"id": cluster, "color": color, "species": []}

        # Get differences for this cluster
        cluster_differences = significant_differences_df.df.filter(
            pl.col("cluster") == cluster
        )

        for row in cluster_differences.iter_rows(named=True):
            taxon_id = row["taxonId"]
            percent_diff = row["percentage_difference"]

            # Get taxonomy info
            taxon_info = taxonomy_df.df.filter(pl.col("taxonId") == taxon_id)
            if (
                taxon_info.height > 0
                and abs(percent_diff) > ClusterSignificantDifferencesDataFrame.THRESHOLD
            ):
                species_data = {
                    "scientific_name": taxon_info["scientificName"][0],
                    "kingdom": taxon_info["kingdom"][0],
                    "taxon_rank": taxon_info["taxonRank"][0],
                    "percent_diff": percent_diff,
                }
                cluster_data["species"].append(species_data)

        clusters_data.append(cluster_data)

    return clusters_data


def prepare_full_report_data(
    cluster_colors_dataframe: ClusterColorDataFrame,
    significant_differences_df: ClusterSignificantDifferencesDataFrame,
    taxonomy_df: TaxonomyDataFrame,
    feature_collection: geojson.FeatureCollection,
) -> dict:
    """
    Prepare data for the full report with maps.

    Args:
        cluster_colors_dataframe: DataFrame with cluster colors
        significant_differences_df: DataFrame with significant taxonomic differences
        taxonomy_df: DataFrame with taxonomy information
        feature_collection: GeoJSON feature collection with cluster boundaries

    Returns:
        Dictionary with all data needed for the report
    """
    clusters_data = []

    # Get unique clusters and sort them
    clusters = sorted(cluster_colors_dataframe.df["cluster"].unique().to_list())

    for cluster in clusters:
        # Get color for this cluster
        color = cluster_colors_dataframe.df.filter(pl.col("cluster") == cluster)[
            "color"
        ].item()

        # Generate the cluster map image
        map_img_base64 = plot_single_cluster(
            feature_collection, cluster, to_base64=True
        )

        cluster_data = {
            "id": cluster,
            "color": color,
            "map_img": map_img_base64,
            "species": [],
        }

        # Get differences for this cluster
        cluster_differences = significant_differences_df.df.filter(
            pl.col("cluster") == cluster
        )

        for row in cluster_differences.iter_rows(named=True):
            taxon_id = row["taxonId"]
            percent_diff = row["percentage_difference"]

            # Get taxonomy info
            taxon_info = taxonomy_df.df.filter(pl.col("taxonId") == taxon_id)
            if (
                taxon_info.height > 0
                and abs(percent_diff) > ClusterSignificantDifferencesDataFrame.THRESHOLD
            ):
                species_data = {
                    "scientific_name": taxon_info["scientificName"][0],
                    "kingdom": taxon_info["kingdom"][0],
                    "taxon_rank": taxon_info["taxonRank"][0],
                    "percent_diff": percent_diff,
                }
                cluster_data["species"].append(species_data)

        clusters_data.append(cluster_data)

    # Generate the overview map
    overview_map_img = plot_entire_region(feature_collection, to_base64=True)

    report_data = {"overview_map": overview_map_img, "clusters": clusters_data}

    return report_data


def render_html(template_name: str, data: dict) -> str:
    """
    Render HTML using Jinja2 templates.

    Args:
        template_name: Name of the template file
        data: Data to pass to the template

    Returns:
        Rendered HTML string
    """
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Set up the Jinja2 environment
    template_dir = os.path.join(current_dir, "..", "templates")
    os.makedirs(template_dir, exist_ok=True)

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(template_dir),
        autoescape=jinja2.select_autoescape(["html", "xml"]),
    )

    template = env.get_template(template_name)
    return template.render(**data)


def write_html(html_content: str, output_file: str) -> None:
    """
    Write HTML content to a file.

    Args:
        html_content: HTML string to write
        output_file: Path to output file
    """
    # Prepare the output file path
    output_file = output.prepare_file_path(output_file)

    with open(output_file, "w") as html_writer:
        html_writer.write(html_content)
