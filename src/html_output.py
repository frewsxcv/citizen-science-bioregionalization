import polars as pl
from src.cluster_stats import Stats
from src.dataframes.geohash_species_counts import GeohashSpeciesCountsDataFrame
from src.dataframes.cluster_color import ClusterColorDataFrame
from src.dataframes.geohash_cluster import GeohashClusterDataFrame


def build_html_output(
    geohash_taxa_counts_dataframe: GeohashSpeciesCountsDataFrame,
    geohash_cluster_dataframe: GeohashClusterDataFrame,
    cluster_colors_dataframe: ClusterColorDataFrame,
    all_stats: Stats,
) -> str:
    html = ""
    for cluster, geohashes, color in (
        geohash_cluster_dataframe
        .df
        .group_by("cluster")
        .agg(pl.col("geohash"))
        .join(cluster_colors_dataframe.df, left_on="cluster", right_on="cluster")
        .iter_rows()
    ):
        html = f"<h1>Cluster {cluster}</h1>"
        html += f"<li>Color: <span style='color: {color};'>{color}</span></li>"
        stats = Stats.build(geohash_taxa_counts_dataframe, geohash_filter=geohashes)

        for kingdom, species, count in (
            stats.taxon.sort(by="count", descending=True)
            .limit(10)
            .select(["kingdom", "species", "count"])
            .collect()
            .iter_rows(named=False)
        ):
            average = (
                stats.taxon.filter(
                    pl.col("kingdom") == kingdom, pl.col("species") == species
                )
                .collect()
                .get_column("average")
                .item()
            )
            all_average = (
                all_stats.taxon.filter(
                    pl.col("kingdom") == kingdom, pl.col("species") == species
                )
                .collect()
                .get_column("average")
                .item()
            )

            # If the difference between the average of the cluster and the average of all is greater than 20%, print it
            percent_diff = (average / all_average * 100) - 100
            if abs(percent_diff) > 20:
                # Print the percentage difference
                html += f"<h2>{species} ({kingdom}):</h2>"
                html += "<ul>"
                html += f"<li>Percentage difference: {'+' if percent_diff > 0 else ''}{percent_diff:.2f}%</li>"
                html += f"<li>Proportion: {average * 100:.2f}%</li>"
                html += f"<li>Count: {count}</li>"
                html += "</ul>"

    return html