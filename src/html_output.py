import polars as pl
from src.dataframes.cluster_color import ClusterColorDataFrame
from src.dataframes.cluster_significant_differences import (
    ClusterSignificantDifferencesDataFrame,
)
from src.dataframes.taxonomy import TaxonomyDataFrame
from src.render import plot_single_cluster, plot_entire_region
import geojson


def build_html_output(
    cluster_colors_dataframe: ClusterColorDataFrame,
    significant_differences_df: ClusterSignificantDifferencesDataFrame,
    taxonomy_df: TaxonomyDataFrame,
) -> str:
    html = ""
    for cluster, color in cluster_colors_dataframe.df.select(
        ["cluster", "color"]
    ).iter_rows(named=False):
        html += f"<h1>Cluster {cluster}</h1>"
        html += f"<li>Color: <span style='color: {color};'>{color}</span></li>"

        # Get differences for this cluster
        cluster_differences = significant_differences_df.df.filter(pl.col("cluster") == cluster)
        
        for row in cluster_differences.iter_rows(named=True):
            taxon_id = row["taxonId"]
            percent_diff = row["percentage_difference"]
            
            # Get taxonomy info
            taxon_info = taxonomy_df.df.filter(pl.col("taxonId") == taxon_id)
            if taxon_info.height > 0:
                # Get taxonomy fields from the first row
                kingdom = taxon_info["kingdom"][0]
                taxon_rank = taxon_info["taxonRank"][0]
                scientific_name = taxon_info["scientificName"][0]
                
                if abs(percent_diff) > ClusterSignificantDifferencesDataFrame.THRESHOLD:
                    html += f"<h2>{scientific_name} ({kingdom}) {taxon_rank}:</h2>"
                    html += "<ul>"
                    html += f"<li>Percentage difference: {'+' if percent_diff > 0 else ''}{percent_diff:.2f}%</li>"
                    html += "</ul>"

    return html


def build_html_output_with_maps(
    cluster_colors_dataframe: ClusterColorDataFrame,
    significant_differences_df: ClusterSignificantDifferencesDataFrame,
    taxonomy_df: TaxonomyDataFrame,
    feature_collection: geojson.FeatureCollection,
) -> str:
    """
    Build HTML output with inline map images for each cluster.
    
    Args:
        cluster_colors_dataframe: DataFrame with cluster colors
        significant_differences_df: DataFrame with significant taxonomic differences
        taxonomy_df: DataFrame with taxonomy information
        feature_collection: GeoJSON feature collection with cluster boundaries
        
    Returns:
        HTML string with inline cluster map images
    """
    html = "<html><head><style>"
    html += """
    body { font-family: Arial, sans-serif; margin: 0; padding: 20px; line-height: 1.6; }
    .cluster-section { margin-bottom: 40px; border: 1px solid #ddd; padding: 20px; border-radius: 5px; }
    .cluster-header { display: flex; align-items: center; margin-bottom: 20px; }
    .cluster-title { margin: 0 20px 0 0; }
    .cluster-map { max-width: 600px; margin-bottom: 20px; border: 1px solid #eee; }
    .full-region-map { max-width: 800px; margin: 20px auto; display: block; border: 1px solid #eee; }
    ul { margin-top: 10px; }
    h2 { margin-top: 20px; margin-bottom: 5px; }
    .color-sample { display: inline-block; width: 20px; height: 20px; margin-right: 10px; border: 1px solid #000; }
    .region-overview { text-align: center; margin-bottom: 30px; }
    """
    html += "</style></head><body>"
    
    html += "<h1>Ecoregion Cluster Analysis</h1>"
    
    # Add overview map of the entire region at the top
    html += '<div class="region-overview">'
    html += '<h2>Complete Ecoregion Map</h2>'
    img_base64 = plot_entire_region(feature_collection, to_base64=True)
    html += f'<img class="full-region-map" src="data:image/png;base64,{img_base64}" alt="Map of All Ecoregion Clusters">'
    html += '</div>'
    
    # Get unique clusters and sort them
    clusters = sorted(cluster_colors_dataframe.df["cluster"].unique().to_list())
    
    for cluster in clusters:
        # Get color for this cluster
        color = cluster_colors_dataframe.df.filter(pl.col("cluster") == cluster)["color"].item()
        
        html += f'<div class="cluster-section">'
        html += f'<div class="cluster-header">'
        html += f'<h1 class="cluster-title">Cluster {cluster}</h1>'
        html += f'<div class="color-sample" style="background-color: {color};"></div>'
        html += f'<span>Color: {color}</span>'
        html += '</div>'
        
        # Generate and embed the cluster map image
        img_base64 = plot_single_cluster(feature_collection, cluster, to_base64=True)
        html += f'<img class="cluster-map" src="data:image/png;base64,{img_base64}" alt="Map of Cluster {cluster}">'
        
        # Get differences for this cluster
        cluster_differences = significant_differences_df.df.filter(pl.col("cluster") == cluster)
        
        if cluster_differences.height > 0:
            html += "<h2>Significant Species</h2>"
            html += "<ul>"
            
            for row in cluster_differences.iter_rows(named=True):
                taxon_id = row["taxonId"]
                percent_diff = row["percentage_difference"]
                
                # Get taxonomy info
                taxon_info = taxonomy_df.df.filter(pl.col("taxonId") == taxon_id)
                if taxon_info.height > 0:
                    # Get taxonomy fields from the first row
                    kingdom = taxon_info["kingdom"][0]
                    taxon_rank = taxon_info["taxonRank"][0]
                    scientific_name = taxon_info["scientificName"][0]
                    
                    if abs(percent_diff) > ClusterSignificantDifferencesDataFrame.THRESHOLD:
                        html += f"<li><strong>{scientific_name}</strong> ({kingdom}, {taxon_rank}): "
                        html += f"{'+' if percent_diff > 0 else ''}{percent_diff:.2f}% difference</li>"
            
            html += "</ul>"
        else:
            html += "<p>No significant species differences found for this cluster.</p>"
        
        html += '</div>'  # close cluster-section
    
    html += "</body></html>"
    return html


def write_html(html_content: str, output_file: str) -> None:
    """
    Write HTML content to a file.
    
    Args:
        html_content: HTML string to write
        output_file: Path to output file
    """
    with open(output_file, "w") as html_writer:
        html_writer.write(html_content)
