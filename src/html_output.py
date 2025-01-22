import polars as pl
from src.cluster_stats import Stats


def build_html_output(cluster_index, clusters, darwin_core_aggregations, all_stats, cluster_colors):
    html = ""
    for cluster, geohashes in cluster_index.iter_clusters_and_geohashes(clusters):
        # Print cluster stats

        color = cluster_colors[cluster]
        html = f"<h1>Cluster {cluster}</h1>"
        html += f"<li>Color: <span style='color: {cluster_colors[cluster]};'>{cluster_colors[cluster]}</span></li>"
        stats = Stats.build(darwin_core_aggregations, geohash_filter=geohashes)

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