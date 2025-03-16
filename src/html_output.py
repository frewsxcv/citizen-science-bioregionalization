import polars as pl
from src.dataframes.cluster_color import ClusterColorDataFrame
from src.dataframes.cluster_significant_differences import (
    ClusterSignificantDifferencesDataFrame,
)
from src.dataframes.taxonomy import TaxonomyDataFrame


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
