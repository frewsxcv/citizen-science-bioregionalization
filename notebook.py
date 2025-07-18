import marimo

__generated_with = "0.14.12"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np

    return mo, np, pl


@app.cell
def _(mo):
    mo.md(r"""# Citizen Science Bioregionalization""")
    return


@app.cell
def _(mo):
    mo.md(r"""## Define inputs""")
    return


@app.cell
def _(mo):
    if mo.running_in_notebook():
        log_file_ui = mo.ui.text("run.log", label="Log file")
        input_file_ui = mo.ui.file_browser(multiple=False, label="Input file")
        geocode_precision_ui = mo.ui.slider(2, 5, value=4, label="Geocode precision")
        taxon_filter_ui = mo.ui.text("", label="Taxon filter (optional)")
        num_clusters_ui = mo.ui.number(value=10, label="Number of clusters")
    else:
        log_file_ui = None
        input_file_ui = None
        geocode_precision_ui = None
        taxon_filter_ui = None
        num_clusters_ui = None

    # Display inputs
    mo.vstack([input_file_ui, geocode_precision_ui, taxon_filter_ui, num_clusters_ui])
    return (
        geocode_precision_ui,
        input_file_ui,
        log_file_ui,
        num_clusters_ui,
        taxon_filter_ui,
    )


@app.cell
def _(
    geocode_precision_ui,
    input_file_ui,
    log_file_ui,
    mo,
    num_clusters_ui,
    taxon_filter_ui,
):
    if mo.running_in_notebook():
        mo.stop(not all([input_file_ui.value]), "Required inputs not inputted")

        log_file = log_file_ui.value
        input_file = str(input_file_ui.path(index=0))
        geocode_precision = geocode_precision_ui.value
        taxon_filter = taxon_filter_ui.value
        num_clusters = num_clusters_ui.value
    else:
        from src.cli_input import parse_cli_input

        cli_input = parse_cli_input()

        log_file = cli_input.log_file
        input_file = cli_input.input_file
        geocode_precision = cli_input.geocode_precision
        taxon_filter = cli_input.taxon_filter
        num_clusters = cli_input.num_clusters
    return geocode_precision, input_file, log_file, num_clusters


@app.cell
def _(mo):
    mo.md(r"""## Set up logging""")
    return


@app.cell
def _(log_file):
    import logging

    logging.basicConfig(filename=log_file, encoding="utf-8", level=logging.INFO)
    return


@app.cell
def _(mo):
    mo.md(r"""## `DarwinCoreCsvLazyFrame`""")
    return


@app.cell
def _(input_file):
    from polars_darwin_core.lf_csv import read_darwin_core_csv

    darwin_core_csv_lazy_frame = read_darwin_core_csv(
        input_file,
        # input_file, taxon_filter=taxon_filter
        # TODO: FIX THE TAXON FILTER ABOVE
    )

    darwin_core_csv_lazy_frame._inner.limit(3).collect()
    return (darwin_core_csv_lazy_frame,)


@app.cell
def _(mo):
    mo.md(r"""## `GeocodeDataFrame`""")
    return


@app.cell
def _(darwin_core_csv_lazy_frame, geocode_precision):
    from src.dataframes.geocode import GeocodeDataFrame

    geocode_dataframe = GeocodeDataFrame.build(
        darwin_core_csv_lazy_frame,
        geocode_precision,
    )

    geocode_dataframe.df
    return (geocode_dataframe,)


@app.cell
def _(mo):
    mo.md(r"""## `TaxonomyDataFrame`""")
    return


@app.cell
def _(darwin_core_csv_lazy_frame):
    from src.dataframes.taxonomy import TaxonomyDataFrame

    taxonomy_dataframe = TaxonomyDataFrame.build(darwin_core_csv_lazy_frame)

    taxonomy_dataframe.df
    return (taxonomy_dataframe,)


@app.cell
def _(mo):
    mo.md(r"""## `GeohashSpeciesCountsDataFrame`""")
    return


@app.cell
def _(darwin_core_csv_lazy_frame, geocode_precision, taxonomy_dataframe):
    from src.dataframes.geocode_taxa_counts import GeocodeTaxaCountsDataFrame

    geocode_taxa_counts_dataframe = GeocodeTaxaCountsDataFrame.build(
        darwin_core_csv_lazy_frame,
        geocode_precision,
        taxonomy_dataframe,
    )

    geocode_taxa_counts_dataframe.df
    return (geocode_taxa_counts_dataframe,)


@app.cell
def _(mo):
    mo.md(r"""## `GeocodeConnectivityMatrix`""")
    return


@app.cell
def _(geocode_dataframe):
    from src.matrices.geocode_connectivity import GeocodeConnectivityMatrix

    geocode_connectivity_matrix = GeocodeConnectivityMatrix.build(geocode_dataframe)

    geocode_connectivity_matrix._connectivity_matrix
    return (geocode_connectivity_matrix,)


@app.cell
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


@app.cell
def _(mo):
    mo.md(r"""## `GeohashClusterDataFrame`""")
    return


@app.cell
def _(mo):
    mo.md(r"""### Build""")
    return


@app.cell
def _(
    geocode_connectivity_matrix,
    geocode_dataframe,
    geocode_distance_matrix,
    num_clusters,
):
    from src.dataframes.geocode_cluster import GeocodeClusterDataFrame

    geocode_cluster_dataframe = GeocodeClusterDataFrame.build(
        geocode_dataframe,
        geocode_distance_matrix,
        geocode_connectivity_matrix,
        num_clusters,
    )
    return (geocode_cluster_dataframe,)


@app.cell
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


@app.cell
def _(mo):
    mo.md(r"""## `ClusterNeighborsDataframe`""")
    return


@app.cell
def _(mo):
    mo.md(r"""### Build""")
    return


@app.cell
def _(geocode_cluster_dataframe, geocode_dataframe):
    from src.dataframes.cluster_neighbors import ClusterNeighborsDataFrame

    cluster_neighbors_dataframe = ClusterNeighborsDataFrame.build(
        geocode_dataframe,
        geocode_cluster_dataframe,
    )
    return (cluster_neighbors_dataframe,)


@app.cell
def _(mo):
    mo.md(r"""### Preview""")
    return


@app.cell
def _(cluster_neighbors_dataframe):
    cluster_neighbors_dataframe.df
    return


@app.cell
def _(mo):
    mo.md(r"""## `ClusterTaxaStatisticsDataFrame`""")
    return


@app.cell
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


@app.cell
def _(mo):
    mo.md(r"""### Preview""")
    return


@app.cell
def _(cluster_taxa_statistics_dataframe):
    cluster_taxa_statistics_dataframe.df
    return


@app.cell
def _(mo):
    mo.md(r"""## `ClusterSignificantDifferencesDataFrame`""")
    return


@app.cell
def _(mo):
    mo.md(r"""### Build""")
    return


@app.cell
def _(cluster_taxa_statistics_dataframe):
    from src.dataframes.cluster_significant_differences import (
        ClusterSignificantDifferencesDataFrame,
    )

    cluster_significant_differences_dataframe = (
        ClusterSignificantDifferencesDataFrame.build(
            cluster_taxa_statistics_dataframe,
        )
    )
    return (cluster_significant_differences_dataframe,)


@app.cell
def _(mo):
    mo.md(r"""### Preview""")
    return


@app.cell
def _(cluster_significant_differences_dataframe):
    cluster_significant_differences_dataframe.df.limit(3)
    return


@app.cell
def _(mo):
    mo.md(r"""## `GeocodeBoundaryDataFrame`""")
    return


@app.cell
def _(mo):
    mo.md(r"""### Build""")
    return


@app.cell
def _(geocode_cluster_dataframe):
    from src.dataframes.geocode_boundary import GeocodeBoundaryDataFrame

    geocode_boundary_dataframe = GeocodeBoundaryDataFrame.build(
        geocode_cluster_dataframe,
    )
    return (geocode_boundary_dataframe,)


@app.cell
def _(mo):
    mo.md(r"""### Preview""")
    return


@app.cell
def _(geocode_boundary_dataframe):
    geocode_boundary_dataframe.df
    return


@app.cell
def _(geocode_boundary_dataframe, pl):
    (
        geocode_boundary_dataframe.df.select(pl.col("geometry"))
        .st.plot(stroke="green")
        .project(type="identity", reflectY=True)
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## `ClusterBoundaryDataFrame`""")
    return


@app.cell
def _(mo):
    mo.md(r"""### Build""")
    return


@app.cell
def _(geocode_boundary_dataframe, geocode_cluster_dataframe):
    from src.dataframes.cluster_boundary import ClusterBoundaryDataFrame

    cluster_boundary_dataframe = ClusterBoundaryDataFrame.build(
        geocode_cluster_dataframe,
        geocode_boundary_dataframe,
    )
    return (cluster_boundary_dataframe,)


@app.cell
def _(mo):
    mo.md(r"""### Preview""")
    return


@app.cell
def _(cluster_boundary_dataframe):
    cluster_boundary_dataframe.df
    return


@app.cell
def _(cluster_boundary_dataframe, pl):
    (
        cluster_boundary_dataframe.df.select(pl.col("geometry"))
        .st.plot(stroke="green")
        .project(type="identity", reflectY=True)
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## `ClusterDistanceMatrix`""")
    return


@app.cell
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


@app.cell
def _(mo):
    mo.md(r"""## `ClusterColorDataFrame`""")
    return


@app.cell
def _(
    cluster_boundary_dataframe,
    cluster_neighbors_dataframe,
    cluster_taxa_statistics_dataframe,
):
    from src.dataframes.cluster_color import ClusterColorDataFrame

    cluster_colors_dataframe = ClusterColorDataFrame.build(
        cluster_neighbors_dataframe,
        cluster_boundary_dataframe,
        cluster_taxa_statistics_dataframe,
        color_method="taxonomic",
        # color_method="geographic",
    )

    cluster_colors_dataframe.df
    return (cluster_colors_dataframe,)


@app.cell
def _(mo):
    mo.md(r"""## `PermanovaResultsDataFrame`""")
    return


@app.cell
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


@app.cell
def _(mo):
    mo.md(r"""### Preview""")
    return


@app.cell
def _(permanova_results_dataframe):
    permanova_results_dataframe.df
    return


@app.cell
def _(mo):
    mo.md(r"""## `GeocodeSilhouetteScoreDataFrame`""")
    return


@app.cell
def _(mo):
    mo.md(r"""### Build""")
    return


@app.cell
def _(
    cluster_neighbors_dataframe,
    geocode_cluster_dataframe,
    geocode_distance_matrix,
):
    from src.dataframes.geocode_silhouette_score import GeocodeSilhouetteScoreDataFrame

    geocode_silhouette_score_dataframe = GeocodeSilhouetteScoreDataFrame.build(
        cluster_neighbors_dataframe,
        geocode_distance_matrix,
        geocode_cluster_dataframe,
    )
    return (geocode_silhouette_score_dataframe,)


@app.cell
def _(mo):
    mo.md(r"""### Preview""")
    return


@app.cell
def _(geocode_silhouette_score_dataframe):
    geocode_silhouette_score_dataframe.df.sort(by="silhouette_score")
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


@app.cell
def _(mo):
    mo.md(r"""## Build and plot GeoJSON feature collection""")
    return


@app.cell
def _(cluster_boundary_dataframe, cluster_colors_dataframe):
    from src.geojson import build_geojson_feature_collection, write_geojson
    from src.render import plot_clusters
    from src import output
    import matplotlib.pyplot as plt

    # Set the figure size
    plt.rcParams["figure.figsize"] = [12, 7]

    feature_collection = build_geojson_feature_collection(
        cluster_boundary_dataframe,
        cluster_colors_dataframe,
    )

    write_geojson(feature_collection, output.get_geojson_path())
    plot_clusters(feature_collection)
    return feature_collection, output


@app.cell
def _(feature_collection):
    import folium

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


@app.cell
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


@app.cell
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


@app.cell
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


@app.cell
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
