import polars as pl
from src.dataframes.cluster_color import ClusterColorDataFrame
from src.dataframes.cluster_significant_differences import (
    ClusterSignificantDifferencesDataFrame,
)


def build_html_output(
    cluster_colors_dataframe: ClusterColorDataFrame,
    significant_differences_df: ClusterSignificantDifferencesDataFrame,
) -> str:
    html = ""
    for cluster, color in cluster_colors_dataframe.df.select(
        ["cluster", "color"]
    ).iter_rows(named=False):
        html += f"<h1>Cluster {cluster}</h1>"
        html += f"<li>Color: <span style='color: {color};'>{color}</span></li>"

        for kingdom, taxonRank, scientificName, percent_diff in (
            significant_differences_df.df.filter(pl.col("cluster") == cluster)
            .select(["kingdom", "taxonRank", "scientificName", "percentage_difference"])
            .iter_rows(named=False)
        ):
            if abs(percent_diff) > ClusterSignificantDifferencesDataFrame.THRESHOLD:
                html += f"<h2>{scientificName} ({kingdom}) {taxonRank}:</h2>"
                html += "<ul>"
                html += f"<li>Percentage difference: {'+' if percent_diff > 0 else ''}{percent_diff:.2f}%</li>"
                html += "</ul>"

    return html
