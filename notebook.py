import marimo

__generated_with = "0.15.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import polars_st as pl_st
    import numpy as np
    import hashlib
    import os
    import polars_darwin_core
    import folium
    return folium, hashlib, mo, np, os, pl, polars_darwin_core


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Citizen Science Bioregionalization""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Define inputs""")
    return


@app.cell
def _(mo):
    log_file_ui = mo.ui.text("run.log", label="Log file")
    input_dir_ui = mo.ui.file_browser(multiple=False, label="Input directory", selection_mode="directory")
    geocode_precision_ui = mo.ui.number(value=4, label="Geocode precision")
    taxon_filter_ui = mo.ui.text("", label="Taxon filter (optional)")
    num_clusters_ui = mo.ui.number(value=10, label="Number of clusters")

    # Display inputs
    (
        mo.vstack(
            [
                log_file_ui,
                input_dir_ui,
                geocode_precision_ui,
                taxon_filter_ui,
                num_clusters_ui,
            ]
        )
        if mo.running_in_notebook()
        else None
    )
    return geocode_precision_ui, input_dir_ui, num_clusters_ui, taxon_filter_ui


@app.cell
def _(geocode_precision_ui, input_dir_ui, num_clusters_ui, taxon_filter_ui):
    import argparse

    parser = argparse.ArgumentParser(
        description="Process Darwin Core CSV data and generate clusters."
    )

    # Add required options
    parser.add_argument(
        "--geocode-precision",
        type=int,
        help="Precision of the geocode",
        default=geocode_precision_ui.value,
    )
    parser.add_argument(
        "--num-clusters",
        type=int,
        help="Number of clusters to generate",
        default=num_clusters_ui.value,
    )
    parser.add_argument("--log-file", type=str, help="Path to the log file")

    # Add optional arguments
    parser.add_argument(
        "--taxon-filter",
        type=str,
        default=taxon_filter_ui.value,
        help="Filter to a specific taxon (e.g., 'Aves')",
    )

    # Positional arguments
    path = str(input_dir_ui.path(index=0)) if input_dir_ui.path(index=0) else None
    parser.add_argument(
        "input_dir", type=str, nargs="?", help="Path to the input directory", default=path
    )

    args = parser.parse_args()
    return (args,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Set up logging""")
    return


@app.cell
def _(args):
    import logging

    logging.basicConfig(filename=args.log_file, encoding="utf-8", level=logging.INFO)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## `DarwinCoreLazyFrame`""")
    return


@app.cell
def _(args, polars_darwin_core):
    darwin_core_csv_lazy_frame = polars_darwin_core.DarwinCoreLazyFrame.from_archive(
        args.input_dir,
        # input_file, taxon_filter=taxon_filter
        # TODO: FIX THE TAXON FILTER ABOVE
    )

    darwin_core_csv_lazy_frame._inner.limit(3).collect()
    return (darwin_core_csv_lazy_frame,)


@app.cell
def _(args, hashlib, os):
    with open(os.path.join(args.input_dir, "occurrence.txt"), "rb") as f:
        file_digest = hashlib.file_digest(f, "sha256").hexdigest()

    output_dir = "tmp"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{file_digest}.parquet")
    return (output_path,)


@app.cell
def _(darwin_core_csv_lazy_frame, os, output_path):
    if not os.path.exists(output_path):
        darwin_core_csv_lazy_frame._inner.sink_parquet(output_path)
    return


@app.cell
def _(output_path, pl, polars_darwin_core):
    inner = pl.scan_parquet(output_path)

    darwin_core_lazy_frame = polars_darwin_core.DarwinCoreLazyFrame(inner)

    darwin_core_lazy_frame._inner.limit(200).collect()
    return (darwin_core_lazy_frame,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## `GeocodeDataFrame`""")
    return


@app.cell
def _(args, darwin_core_lazy_frame):
    from src.dataframes.geocode import GeocodeDataFrame

    geocode_dataframe = GeocodeDataFrame.build(
        darwin_core_lazy_frame,
        args.geocode_precision,
    )

    geocode_dataframe.df
    return (geocode_dataframe,)


@app.cell(hide_code=True)
def _(folium, geocode_dataframe, pl):
    _center = geocode_dataframe.df.select(
        pl.col("center").alias("geometry"),
    )
    _boundary = geocode_dataframe.df.select(
        pl.col("boundary").alias("geometry"),
    )

    _map = folium.Map(
        tiles="Esri.WorldGrayCanvas",
    )

    folium.GeoJson(
        _center.st,
        marker=folium.Circle(),
    ).add_to(_map)

    folium.GeoJson(
        _boundary.st,
    ).add_to(_map)

    _map.fit_bounds(_map.get_bounds())

    _map
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## `TaxonomyDataFrame`""")
    return


@app.cell
def _(darwin_core_lazy_frame):
    from src.dataframes.taxonomy import TaxonomyDataFrame

    taxonomy_dataframe = TaxonomyDataFrame.build(darwin_core_lazy_frame)

    taxonomy_dataframe.df
    return (taxonomy_dataframe,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## `GeocodeSpeciesCountsDataFrame`""")
    return


@app.cell
def _(args, darwin_core_lazy_frame, taxonomy_dataframe):
    from src.dataframes.geocode_taxa_counts import GeocodeTaxaCountsDataFrame

    geocode_taxa_counts_dataframe = GeocodeTaxaCountsDataFrame.build(
        darwin_core_lazy_frame,
        args.geocode_precision,
        taxonomy_dataframe,
    )

    geocode_taxa_counts_dataframe.df
    return (geocode_taxa_counts_dataframe,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## `GeocodeConnectivityMatrix`""")
    return


@app.cell
def _(geocode_dataframe):
    from src.matrices.geocode_connectivity import GeocodeConnectivityMatrix

    geocode_connectivity_matrix = GeocodeConnectivityMatrix.build(geocode_dataframe)

    geocode_connectivity_matrix._connectivity_matrix
    return (geocode_connectivity_matrix,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## `GeocodeDistanceMatrix`""")
    return


@app.cell
def _(geocode_dataframe, geocode_taxa_counts_dataframe, mo, np):
    from src.matrices.geocode_distance import GeocodeDistanceMatrix

    geocode_distance_matrix = GeocodeDistanceMatrix.build(
        geocode_taxa_counts_dataframe,
        geocode_dataframe,
    )

    mo.vstack(
        [
            mo.md(GeocodeDistanceMatrix.__doc__),
            mo.plain_text(np.array_repr(geocode_distance_matrix.squareform())),
        ]
    )
    return (geocode_distance_matrix,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## `GeocodeClusterDataFrame`""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Build""")
    return


@app.cell
def _(
    args,
    geocode_connectivity_matrix,
    geocode_dataframe,
    geocode_distance_matrix,
):
    from src.dataframes.geocode_cluster import GeocodeClusterDataFrame

    geocode_cluster_dataframe = GeocodeClusterDataFrame.build(
        geocode_dataframe,
        geocode_distance_matrix,
        geocode_connectivity_matrix,
        args.num_clusters,
    )
    return (geocode_cluster_dataframe,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Preview""")
    return


@app.cell
def _(geocode_cluster_dataframe):
    geocode_cluster_dataframe.df.limit(3)
    return


@app.cell
def _():
    # # TMP

    # from src.dataframes.geocode_cluster import GeocodeClusterDataFrame
    # from sklearn.metrics import silhouette_score

    # results = []

    # for i in range(2, 200):
    #     geocode_cluster_dataframe = GeocodeClusterDataFrame.build(
    #         geocode_dataframe,
    #         distance_matrix,
    #         connectivity_matrix,
    #         num_clusters=i,
    #     )
    #     score = silhouette_score(
    #         X=distance_matrix.squareform(),
    #         labels=geocode_cluster_dataframe.df["cluster"],
    #         metric="precomputed",
    #     )
    #     print(f"{i}: {score}")
    #     results.append((i, score))

    # results
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## `ClusterNeighborsDataframe`""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Build""")
    return


@app.cell
def _(geocode_cluster_dataframe, geocode_dataframe):
    from src.dataframes.cluster_neighbors import ClusterNeighborsSchema

    cluster_neighbors_dataframe = ClusterNeighborsSchema.build(
        geocode_dataframe,
        geocode_cluster_dataframe,
    )
    return (cluster_neighbors_dataframe,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Preview""")
    return


@app.cell
def _(cluster_neighbors_dataframe):
    cluster_neighbors_dataframe
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## `ClusterTaxaStatisticsDataFrame`""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Build""")
    return


@app.cell
def _(
    geocode_cluster_dataframe,
    geocode_taxa_counts_dataframe,
    taxonomy_dataframe,
):
    from src.dataframes.cluster_taxa_statistics import ClusterTaxaStatisticsDataFrame

    cluster_taxa_statistics_dataframe = ClusterTaxaStatisticsDataFrame.build(
        geocode_taxa_counts_dataframe,
        geocode_cluster_dataframe,
        taxonomy_dataframe,
    )
    return (cluster_taxa_statistics_dataframe,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Preview""")
    return


@app.cell
def _(cluster_taxa_statistics_dataframe):
    cluster_taxa_statistics_dataframe.df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## `ClusterSignificantDifferencesDataFrame`""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Build""")
    return


@app.cell
def _(cluster_taxa_statistics_dataframe):
    from src.dataframes.cluster_significant_differences import (
        ClusterSignificantDifferencesSchema,
    )

    cluster_significant_differences_dataframe = (
        ClusterSignificantDifferencesSchema.build(
            cluster_taxa_statistics_dataframe,
        )
    )
    return (cluster_significant_differences_dataframe,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Preview""")
    return


@app.cell
def _(cluster_significant_differences_dataframe):
    cluster_significant_differences_dataframe
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## `ClusterBoundarySchema`""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Build""")
    return


@app.cell
def _(geocode_cluster_dataframe, geocode_dataframe):
    from src.dataframes.cluster_boundary import ClusterBoundarySchema

    cluster_boundary_dataframe = ClusterBoundarySchema.build(
        geocode_cluster_dataframe,
        geocode_dataframe,
    )
    return (cluster_boundary_dataframe,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Preview""")
    return


@app.cell
def _(cluster_boundary_dataframe):
    cluster_boundary_dataframe
    return


@app.cell
def _(cluster_boundary_dataframe, pl):
    (
        cluster_boundary_dataframe.select(pl.col("geometry"))
        .st.plot(stroke="green")
        .project(type="identity", reflectY=True)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## `ClusterDistanceMatrix`""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Build""")
    return


@app.cell
def _(cluster_taxa_statistics_dataframe):
    from src.matrices.cluster_distance import ClusterDistanceMatrix

    cluster_distance_matrix = ClusterDistanceMatrix.build(
        cluster_taxa_statistics_dataframe,
    )
    return (cluster_distance_matrix,)


@app.cell
def _(mo):
    mo.md(r"""### Preview""")
    return


@app.cell
def _(cluster_distance_matrix):
    cluster_distance_matrix.squareform()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## `ClusterColorDataFrame`""")
    return


@app.cell
def _(
    cluster_boundary_dataframe,
    cluster_neighbors_dataframe,
    cluster_taxa_statistics_dataframe,
):
    from src.dataframes.cluster_color import ClusterColorSchema

    cluster_colors_dataframe = ClusterColorSchema.build(
        cluster_neighbors_dataframe,
        cluster_boundary_dataframe,
        cluster_taxa_statistics_dataframe,
        color_method="taxonomic",
        # color_method="geographic",
    )

    cluster_colors_dataframe
    return (cluster_colors_dataframe,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## `PermanovaResultsDataFrame`""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Build""")
    return


@app.cell
def _(geocode_cluster_dataframe, geocode_dataframe, geocode_distance_matrix):
    from src.dataframes.permanova_results import PermanovaResultsDataFrame

    permanova_results_dataframe = PermanovaResultsDataFrame.build(
        geocode_distance_matrix=geocode_distance_matrix,
        geocode_cluster_dataframe=geocode_cluster_dataframe,
        geocode_dataframe=geocode_dataframe,
    )
    return (permanova_results_dataframe,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Preview""")
    return


@app.cell
def _(permanova_results_dataframe):
    permanova_results_dataframe.df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## `GeocodeSilhouetteScoreDataFrame`""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Build""")
    return


@app.cell
def _(
    cluster_neighbors_dataframe,
    geocode_cluster_dataframe,
    geocode_distance_matrix,
):
    from src.dataframes.geocode_silhouette_score import GeocodeSilhouetteScoreSchema

    geocode_silhouette_score_dataframe = GeocodeSilhouetteScoreSchema.build(
        cluster_neighbors_dataframe,
        geocode_distance_matrix,
        geocode_cluster_dataframe,
    )
    return (geocode_silhouette_score_dataframe,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Preview""")
    return


@app.cell
def _(geocode_silhouette_score_dataframe):
    geocode_silhouette_score_dataframe.sort(by="silhouette_score")
    return


@app.cell
def _(
    cluster_colors_dataframe,
    geocode_cluster_dataframe,
    geocode_distance_matrix,
    geocode_silhouette_score_dataframe,
):
    from src.plot.silhouette_score import plot_silhouette_scores

    plot_silhouette_scores(
        geocode_cluster_dataframe,
        geocode_distance_matrix,
        geocode_silhouette_score_dataframe,
        cluster_colors_dataframe,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Build and plot GeoJSON feature collection""")
    return


@app.cell
def _(cluster_boundary_dataframe, cluster_colors_dataframe):
    from src.geojson import build_geojson_feature_collection

    feature_collection = build_geojson_feature_collection(
        cluster_boundary_dataframe,
        cluster_colors_dataframe,
    )
    return (feature_collection,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Save""")
    return


@app.cell
def _(feature_collection):
    from src.geojson import write_geojson
    from src import output

    write_geojson(feature_collection, output.get_geojson_path())
    return (output,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Plot""")
    return


@app.cell
def _(feature_collection, folium):
    map = folium.Map(
        tiles="Esri.WorldGrayCanvas",
    )

    folium.GeoJson(
        feature_collection,
        style_function=lambda feature: feature["properties"],
    ).add_to(map)

    map.fit_bounds(folium.utilities.get_bounds(feature_collection, lonlat=True))

    map
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Build and display HTML output""")
    return


@app.cell
def _(
    cluster_colors_dataframe,
    cluster_significant_differences_dataframe,
    feature_collection,
    output,
    taxonomy_dataframe,
):
    from src.html_output import prepare_full_report_data, render_html, write_html

    report_data = prepare_full_report_data(
        cluster_colors_dataframe,
        cluster_significant_differences_dataframe,
        taxonomy_dataframe,
        feature_collection,
    )
    html_content = render_html("cluster_report.html", report_data)
    html_output = output.get_html_path()
    write_html(html_content, html_output)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Dimensionality reduction plot""")
    return


@app.cell
def _(
    cluster_colors_dataframe,
    geocode_cluster_dataframe,
    geocode_distance_matrix,
):
    from src.plot.dimnesionality_reduction import create_dimensionality_reduction_plot

    create_dimensionality_reduction_plot(
        geocode_distance_matrix,
        geocode_cluster_dataframe,
        cluster_colors_dataframe,
        method="umap",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Clustermap visualization""")
    return


@app.cell
def _(
    cluster_colors_dataframe,
    cluster_significant_differences_dataframe,
    cluster_taxa_statistics_dataframe,
    geocode_cluster_dataframe,
    geocode_dataframe,
    geocode_distance_matrix,
    geocode_taxa_counts_dataframe,
    taxonomy_dataframe,
):
    from src.plot.cluster_taxa import create_cluster_taxa_heatmap

    create_cluster_taxa_heatmap(
        geocode_dataframe=geocode_dataframe,
        geocode_cluster_dataframe=geocode_cluster_dataframe,
        cluster_colors_dataframe=cluster_colors_dataframe,
        geocode_distance_matrix=geocode_distance_matrix,
        cluster_significant_differences_dataframe=cluster_significant_differences_dataframe,
        taxonomy_dataframe=taxonomy_dataframe,
        geocode_taxa_counts_dataframe=geocode_taxa_counts_dataframe,
        cluster_taxa_statistics_dataframe=cluster_taxa_statistics_dataframe,
        limit_species=5,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Data flow""")
    return


@app.cell
def _(mo):
    from src.dependency_graph import plot_dependency_graph

    mo.mermaid(plot_dependency_graph())
    return


if __name__ == "__main__":
    app.run()
