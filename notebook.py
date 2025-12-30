# pyright: reportUnusedExpression=false

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import folium
    import marimo as mo
    import numpy as np
    import polars as pl

    from src.cache_parquet import cache_parquet
    from src.types import Bbox

    return Bbox, cache_parquet, folium, mo, np, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Citizen Science Bioregionalization
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Define inputs
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    log_file_ui = mo.ui.text("run.log", label="Log file")
    parquet_source_path_ui = mo.ui.text(
        "gs://public-datasets-gbif/occurrence/2025-11-01/occurrence.parquet/*",
        label="Input GCS directory",
    )
    geocode_precision_ui = mo.ui.number(value=4, label="Geocode precision")
    min_clusters_to_test_ui = mo.ui.number(value=2, label="Min clusters to test")
    max_clusters_to_test_ui = mo.ui.number(value=20, label="Max clusters to test")
    taxon_filter_ui = mo.ui.text("", label="Taxon filter (optional)")
    limit_results_enabled_ui = mo.ui.checkbox(value=True, label="Enable limit")
    limit_results_value_ui = mo.ui.number(value=1000, label="Limit results")
    min_lon_ui = mo.ui.number(value=-87.0, label="Min Longitude")
    min_lat_ui = mo.ui.number(value=25.0, label="Min Latitude")
    max_lon_ui = mo.ui.number(value=-66.0, label="Max Longitude")
    max_lat_ui = mo.ui.number(value=47.0, label="Max Latitude")
    run_button_ui = mo.ui.run_button()
    return (
        geocode_precision_ui,
        limit_results_enabled_ui,
        limit_results_value_ui,
        log_file_ui,
        max_clusters_to_test_ui,
        max_lat_ui,
        max_lon_ui,
        min_clusters_to_test_ui,
        min_lat_ui,
        min_lon_ui,
        parquet_source_path_ui,
        run_button_ui,
        taxon_filter_ui,
    )


@app.cell(hide_code=True)
def _(log_file_ui):
    log_file_ui
    return


@app.cell(hide_code=True)
def _(parquet_source_path_ui):
    parquet_source_path_ui
    return


@app.cell(hide_code=True)
def _(geocode_precision_ui):
    geocode_precision_ui
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Cluster Configuration

    Number of clusters will be automatically optimized based on silhouette scores.
    """)
    return


@app.cell(hide_code=True)
def _(max_clusters_to_test_ui, min_clusters_to_test_ui, mo):
    mo.vstack(
        [
            min_clusters_to_test_ui,
            max_clusters_to_test_ui,
        ]
    )
    return


@app.cell(hide_code=True)
def _(taxon_filter_ui):
    taxon_filter_ui
    return


@app.cell(hide_code=True)
def _(limit_results_enabled_ui, limit_results_value_ui, mo):
    mo.vstack(
        [
            limit_results_enabled_ui,
            limit_results_value_ui,
        ]
    )
    return


@app.cell(hide_code=True)
def _(min_lat_ui, min_lon_ui, mo):
    mo.vstack([min_lon_ui, min_lat_ui])
    return


@app.cell(hide_code=True)
def _(max_lat_ui, max_lon_ui, mo):
    mo.vstack([max_lon_ui, max_lat_ui])
    return


@app.cell(hide_code=True)
def _(folium, max_lat_ui, max_lon_ui, min_lat_ui, min_lon_ui):
    _map = folium.Map(
        tiles="Esri.WorldGrayCanvas",
    )

    _bounds = [
        [min_lat_ui.value, min_lon_ui.value],
        [max_lat_ui.value, max_lon_ui.value],
    ]

    folium.Rectangle(bounds=_bounds).add_to(_map)

    _map.fit_bounds(_bounds, padding=[20, 20])

    _map
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Set up CLI
    """)
    return


@app.cell(hide_code=True)
def _(
    geocode_precision_ui,
    limit_results_enabled_ui,
    limit_results_value_ui,
    log_file_ui,
    max_clusters_to_test_ui,
    max_lat_ui,
    max_lon_ui,
    min_clusters_to_test_ui,
    min_lat_ui,
    min_lon_ui,
    parquet_source_path_ui,
    taxon_filter_ui,
):
    from src.cli import parse_args_with_defaults

    args = parse_args_with_defaults(
        geocode_precision=geocode_precision_ui.value,
        taxon_filter=taxon_filter_ui.value,
        min_lat=min_lat_ui.value,
        max_lat=max_lat_ui.value,
        min_lon=min_lon_ui.value,
        max_lon=max_lon_ui.value,
        limit_results=limit_results_value_ui.value
        if limit_results_enabled_ui.value
        else None,
        parquet_source_path=parquet_source_path_ui.value,
        log_file=log_file_ui.value,
        min_clusters_to_test=min_clusters_to_test_ui.value,
        max_clusters_to_test=max_clusters_to_test_ui.value,
    )
    return (args,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Start notebook
    """)
    return


@app.cell(hide_code=True)
def _(args, mo, run_button_ui):
    limit_results = args.limit_results
    no_stop = args.no_stop
    log_file = args.log_file
    parquet_source_path = args.parquet_source_path
    min_lat = args.min_lat
    max_lat = args.max_lat
    min_lon = args.min_lon
    max_lon = args.max_lon
    taxon_filter = args.taxon_filter
    geocode_precision = args.geocode_precision
    min_clusters_to_test = args.min_clusters
    max_clusters_to_test = args.max_clusters

    inputs_table = mo.ui.table(
        label="Inputs",
        selection=None,
        data=[
            {"variable": "limit_results", "value": limit_results},
            {"variable": "log_file", "value": log_file},
            {"variable": "parquet_source_path", "value": parquet_source_path},
            {"variable": "min_lat", "value": min_lat},
            {"variable": "max_lat", "value": max_lat},
            {"variable": "min_lon", "value": min_lon},
            {"variable": "max_lon", "value": max_lon},
            {"variable": "taxon_filter", "value": taxon_filter},
            {"variable": "geocode_precision", "value": geocode_precision},
            {"variable": "min_clusters_to_test", "value": min_clusters_to_test},
            {"variable": "max_clusters_to_test", "value": max_clusters_to_test},
        ],
    )

    output2 = mo.vstack(
        [
            inputs_table,
            run_button_ui,
        ]
    )

    if mo.running_in_notebook() and not no_stop:
        mo.stop(not run_button_ui.value, output2)

    mo.md("Notebook started")
    return (
        geocode_precision,
        limit_results,
        log_file,
        max_clusters_to_test,
        max_lat,
        max_lon,
        min_clusters_to_test,
        min_lat,
        min_lon,
        parquet_source_path,
        taxon_filter,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Set up logging
    """)
    return


@app.cell(hide_code=True)
def _(log_file):
    import logging

    logging.basicConfig(filename=log_file, encoding="utf-8", level=logging.INFO)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## `DarwinCore`
    """)
    return


@app.cell
def _(
    Bbox,
    limit_results,
    max_lat,
    max_lon,
    min_lat,
    min_lon,
    parquet_source_path,
    taxon_filter,
):
    from src.dataframes.darwin_core import DarwinCoreSchema

    darwin_core_lf = DarwinCoreSchema.build_lf(
        source_path=parquet_source_path,
        bounding_box=Bbox.from_coordinates(min_lat, max_lat, min_lon, max_lon),
        limit=limit_results,
        taxon_filter=taxon_filter,
    )
    return (darwin_core_lf,)


@app.cell
def _(darwin_core_lf, pl):
    darwin_core_lf.select(pl.len()).collect(engine="streaming")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## `Geocode`
    """)
    return


@app.cell
def _(
    Bbox,
    cache_parquet,
    darwin_core_lf,
    geocode_precision,
    max_lat,
    max_lon,
    min_lat,
    min_lon,
):
    from src.dataframes.geocode import GeocodeNoEdgesSchema, GeocodeSchema

    geocode_lf_with_edges = cache_parquet(
        GeocodeSchema.build_df(
            darwin_core_lf,
            geocode_precision,
            bounding_box=Bbox.from_coordinates(min_lat, max_lat, min_lon, max_lon),
        ),
        cache_key=GeocodeSchema,
    )

    geocode_lf = cache_parquet(
        GeocodeNoEdgesSchema.from_geocode_schema(
            geocode_lf_with_edges,
        ),
        cache_key=GeocodeNoEdgesSchema,
    )
    return geocode_lf, geocode_lf_with_edges


@app.cell(hide_code=True)
def _(folium, geocode_lf, geocode_lf_with_edges, pl):
    _center = geocode_lf.select(
        pl.col("center").alias("geometry"),
    ).collect()
    _boundary = geocode_lf_with_edges.select(
        pl.col("boundary").alias("geometry"),
        pl.col("is_edge"),
    ).collect()

    _map = folium.Map(
        tiles="Esri.WorldGrayCanvas",
    )

    folium.GeoJson(
        _center.st,
        marker=folium.Circle(),
    ).add_to(_map)

    def style(n):
        return {"color": "grey" if n["properties"]["is_edge"] else "blue"}

    folium.GeoJson(_boundary.st, style_function=style).add_to(_map)

    _map.fit_bounds(_map.get_bounds())

    _map
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## `Taxonomy`
    """)
    return


@app.cell
def _(
    Bbox,
    cache_parquet,
    darwin_core_lf,
    geocode_lf,
    geocode_precision,
    max_lat,
    max_lon,
    min_lat,
    min_lon,
):
    from src.dataframes.taxonomy import TaxonomySchema

    taxonomy_lf = cache_parquet(
        TaxonomySchema.build_lf(
            darwin_core_lf,
            geocode_precision,
            geocode_lf,
            bounding_box=Bbox.from_coordinates(min_lat, max_lat, min_lon, max_lon),
        ),
        cache_key=TaxonomySchema,
    )
    return (taxonomy_lf,)


@app.cell
def _(taxonomy_lf):
    taxonomy_lf.limit(100).collect(engine="streaming")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## `GeocodeTaxaCounts`
    """)
    return


@app.cell
def _(
    Bbox,
    cache_parquet,
    darwin_core_lf,
    geocode_lf,
    geocode_precision,
    max_lat,
    max_lon,
    min_lat,
    min_lon,
    taxonomy_lf,
):
    from src.dataframes.geocode_taxa_counts import GeocodeTaxaCountsSchema

    geocode_taxa_counts_lf = cache_parquet(
        GeocodeTaxaCountsSchema.build_df(
            darwin_core_lf,
            geocode_precision,
            taxonomy_lf,
            geocode_lf,
            bounding_box=Bbox.from_coordinates(min_lat, max_lat, min_lon, max_lon),
        ),
        cache_key=GeocodeTaxaCountsSchema,
    )
    return (geocode_taxa_counts_lf,)


@app.cell
def _(geocode_taxa_counts_lf):
    geocode_taxa_counts_lf.limit(100).collect(engine="streaming")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## `GeocodeConnectivity`
    """)
    return


@app.cell
def _(geocode_lf):
    from src.matrices.geocode_connectivity import GeocodeConnectivityMatrix

    geocode_connectivity_matrix = GeocodeConnectivityMatrix.build(geocode_lf)

    geocode_connectivity_matrix._connectivity_matrix
    return (geocode_connectivity_matrix,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## `GeocodeDistance`
    """)
    return


@app.cell
def _(geocode_lf, geocode_taxa_counts_lf, mo, np):
    from src.matrices.geocode_distance import GeocodeDistanceMatrix

    geocode_distance_matrix = GeocodeDistanceMatrix.build(
        geocode_taxa_counts_lf,
        geocode_lf,
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
    mo.md(r"""
    ## Cluster Optimization
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Build Clustering for All K Values
    """)
    return


@app.cell
def _(
    cache_parquet,
    geocode_connectivity_matrix,
    geocode_distance_matrix,
    geocode_lf,
    max_clusters_to_test,
    min_clusters_to_test,
):
    from src.dataframes.geocode_cluster import GeocodeClusterMultiKSchema

    all_clusters_df = cache_parquet(
        GeocodeClusterMultiKSchema.build_df(
            geocode_lf,
            geocode_distance_matrix,
            geocode_connectivity_matrix,
            min_k=min_clusters_to_test,
            max_k=max_clusters_to_test,
        ),
        cache_key=GeocodeClusterMultiKSchema,
    ).collect(engine="streaming")
    return (all_clusters_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Find Optimal K
    """)
    return


@app.cell
def _(all_clusters_df, cache_parquet, geocode_distance_matrix, mo):
    from src.cluster_optimization import optimize_num_clusters

    optimal_num_clusters, all_silhouette_scores = optimize_num_clusters(
        geocode_distance_matrix,
        all_clusters_df,
    )

    # Cache the results
    from src.cluster_optimization import optimize_num_clusters as optimization_cache_key

    all_silhouette_scores_df = cache_parquet(
        all_silhouette_scores,
        cache_key=optimization_cache_key,
    ).collect(engine="streaming")

    # Create base GeocodeClusterSchema (single k) for downstream use
    from src.dataframes.geocode_cluster import GeocodeClusterSchema

    geocode_cluster_df = cache_parquet(
        GeocodeClusterSchema.from_multi_k(
            all_clusters_df,
            optimal_num_clusters,
        ),
        cache_key=GeocodeClusterSchema,
    ).collect(engine="streaming")

    mo.md(f"**Optimal number of clusters: k={optimal_num_clusters}**")
    return all_silhouette_scores_df, geocode_cluster_df, optimal_num_clusters


@app.cell
def _(all_silhouette_scores_df):
    all_silhouette_scores_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Optimization Results Table
    """)
    return


@app.cell
def _(all_silhouette_scores_df):
    from src.cluster_optimization import format_optimization_results

    format_optimization_results(all_silhouette_scores_df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Silhouette Score vs Number of Clusters
    """)
    return


@app.cell
def _(all_silhouette_scores_df, optimal_num_clusters):
    from src.plot.cluster_optimization import plot_silhouette_vs_k

    plot_silhouette_vs_k(
        all_silhouette_scores_df,
        optimal_k=optimal_num_clusters,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Silhouette Score Distributions
    """)
    return


@app.cell
def _(all_silhouette_scores_df):
    from src.plot.cluster_optimization import plot_silhouette_distributions

    plot_silhouette_distributions(
        all_silhouette_scores_df,
        top_n=5,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## `ClusterNeighbors`
    """)
    return


@app.cell
def _(cache_parquet, geocode_cluster_df, geocode_lf):
    from src.dataframes.cluster_neighbors import ClusterNeighborsSchema

    cluster_neighbors_lf = cache_parquet(
        ClusterNeighborsSchema.build_df(
            geocode_lf,
            geocode_cluster_df,
        ),
        cache_key=ClusterNeighborsSchema,
    )
    return (cluster_neighbors_lf,)


@app.cell
def _(cluster_neighbors_lf):
    cluster_neighbors_lf.limit(3).collect()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## `ClusterTaxaStatistics`
    """)
    return


@app.cell
def _(cache_parquet, geocode_cluster_df, geocode_taxa_counts_lf, taxonomy_lf):
    from src.dataframes.cluster_taxa_statistics import ClusterTaxaStatisticsSchema

    cluster_taxa_statistics_df = cache_parquet(
        ClusterTaxaStatisticsSchema.build_df(
            geocode_taxa_counts_lf,
            geocode_cluster_df.lazy(),
            taxonomy_lf,
        ),
        cache_key=ClusterTaxaStatisticsSchema,
    ).collect(engine="streaming")
    return (cluster_taxa_statistics_df,)


@app.cell
def _(cluster_taxa_statistics_df):
    cluster_taxa_statistics_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## `ClusterSignificantDifferences`
    """)
    return


@app.cell
def _(cache_parquet, cluster_neighbors_lf, cluster_taxa_statistics_df):
    from src.dataframes.cluster_significant_differences import (
        ClusterSignificantDifferencesSchema,
    )

    cluster_significant_differences_df = cache_parquet(
        ClusterSignificantDifferencesSchema.build_df(
            cluster_taxa_statistics_df,
            cluster_neighbors_lf,
        ),
        cache_key=ClusterSignificantDifferencesSchema,
    ).collect(engine="streaming")
    return (cluster_significant_differences_df,)


@app.cell
def _(cluster_significant_differences_df):
    cluster_significant_differences_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## `ClusterBoundary`
    """)
    return


@app.cell
def _(cache_parquet, geocode_cluster_df, geocode_lf):
    from src.dataframes.cluster_boundary import ClusterBoundarySchema

    cluster_boundary_df = cache_parquet(
        ClusterBoundarySchema.build_df(
            geocode_cluster_df,
            geocode_lf,
        ),
        cache_key=ClusterBoundarySchema,
    ).collect(engine="streaming")
    return (cluster_boundary_df,)


@app.cell
def _(cluster_boundary_df):
    cluster_boundary_df
    return


@app.cell(hide_code=True)
def _(cluster_boundary_df, folium):
    _boundary = cluster_boundary_df.select(["geometry", "cluster"])

    _map = folium.Map(
        tiles="Esri.WorldGrayCanvas",
    )

    folium.GeoJson(
        _boundary.st,
        popup=folium.GeoJsonPopup(
            fields=["cluster"],
        ),
    ).add_to(_map)

    _map.fit_bounds(_map.get_bounds())

    _map
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## `ClusterDistance`
    """)
    return


@app.cell
def _(cluster_taxa_statistics_df):
    from src.matrices.cluster_distance import ClusterDistanceMatrix

    cluster_distance_matrix = ClusterDistanceMatrix.build(
        cluster_taxa_statistics_df,
    )
    return (cluster_distance_matrix,)


@app.cell
def _(cluster_distance_matrix):
    cluster_distance_matrix.squareform()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## `ClusterColor`
    """)
    return


@app.cell
def _(
    cache_parquet,
    cluster_boundary_df,
    cluster_neighbors_lf,
    cluster_taxa_statistics_df,
    optimal_num_clusters,
):
    from src.dataframes.cluster_color import ClusterColorSchema

    # Use taxonomic coloring if we have at least 10 clusters, otherwise use geographic
    color_method = "taxonomic" if optimal_num_clusters >= 10 else "geographic"

    cluster_colors_df = cache_parquet(
        ClusterColorSchema.build_df(
            cluster_neighbors_lf,
            cluster_boundary_df,
            cluster_taxa_statistics_df,
            color_method=color_method,
        ),
        cache_key=ClusterColorSchema,
    ).collect(engine="streaming")
    return (cluster_colors_df,)


@app.cell
def _(cluster_colors_df):
    cluster_colors_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## `PermanovaResults`
    """)
    return


@app.cell
def _(cache_parquet, geocode_cluster_df, geocode_distance_matrix, geocode_lf):
    from src.dataframes.permanova_results import PermanovaResultsSchema

    permanova_results_df = cache_parquet(
        PermanovaResultsSchema.build_df(
            geocode_distance_matrix=geocode_distance_matrix,
            geocode_cluster_df=geocode_cluster_df,
            geocode_lf=geocode_lf,
        ),
        cache_key=PermanovaResultsSchema,
    ).collect(engine="streaming")
    return (permanova_results_df,)


@app.cell
def _(permanova_results_df):
    permanova_results_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## `GeocodeSilhouetteScore`
    """)
    return


@app.cell
def _(all_silhouette_scores_df, optimal_num_clusters, pl):
    # Filter the already-computed silhouette scores to just the optimal k
    geocode_silhouette_score_df = all_silhouette_scores_df.filter(
        pl.col("num_clusters") == optimal_num_clusters
    )
    return (geocode_silhouette_score_df,)


@app.cell
def _(geocode_silhouette_score_df):
    geocode_silhouette_score_df.sort(by="silhouette_score")
    return


@app.cell
def _(
    cluster_colors_df,
    geocode_cluster_df,
    geocode_distance_matrix,
    geocode_silhouette_score_df,
):
    from src.plot.silhouette_score import plot_silhouette_scores

    plot_silhouette_scores(
        geocode_cluster_df,
        geocode_distance_matrix,
        geocode_silhouette_score_df,
        cluster_colors_df,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Build and plot GeoJSON feature collection
    """)
    return


@app.cell
def _(cluster_boundary_df, cluster_colors_df):
    from src.geojson import build_geojson_feature_collection

    feature_collection = build_geojson_feature_collection(
        cluster_boundary_df,
        cluster_colors_df,
    )
    return (feature_collection,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Save
    """)
    return


@app.cell
def _(feature_collection):
    from src import output
    from src.geojson import write_geojson

    write_geojson(feature_collection, output.get_geojson_path())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Plot
    """)
    return


@app.cell
def _(feature_collection, folium):
    _map = folium.Map(
        tiles="Esri.WorldGrayCanvas",
    )

    folium.GeoJson(
        feature_collection,
        style_function=lambda feature: feature["properties"],
    ).add_to(_map)

    _map.fit_bounds(folium.utilities.get_bounds(feature_collection, lonlat=True))

    _map
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dimensionality reduction plot
    """)
    return


@app.cell
def _(cluster_colors_df, geocode_cluster_df, geocode_distance_matrix):
    from src.plot.dimensionality_reduction import create_dimensionality_reduction_plot

    _chart = create_dimensionality_reduction_plot(
        geocode_distance_matrix,
        geocode_cluster_df,
        cluster_colors_df,
        method="umap",
    )
    _chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Clustermap visualization
    """)
    return


@app.cell
def _(
    cluster_colors_df,
    cluster_significant_differences_df,
    cluster_taxa_statistics_df,
    geocode_cluster_df,
    geocode_distance_matrix,
    geocode_lf,
    geocode_taxa_counts_lf,
    mo,
    taxonomy_lf,
):
    from src.plot.cluster_taxa import create_cluster_taxa_heatmap

    heatmap = create_cluster_taxa_heatmap(
        geocode_lf=geocode_lf,
        geocode_cluster_df=geocode_cluster_df,
        cluster_colors_df=cluster_colors_df,
        geocode_distance_matrix=geocode_distance_matrix,
        cluster_significant_differences_df=cluster_significant_differences_df,
        taxonomy_df=taxonomy_lf.collect(engine="streaming"),
        geocode_taxa_counts_lf=geocode_taxa_counts_lf,
        cluster_taxa_statistics_df=cluster_taxa_statistics_df,
        limit_species=5,
    )

    if heatmap is None:
        result = mo.md("No significant differences found between clusters.")
    else:
        result = heatmap.figure

    result
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## `SignificantTaxaImages`
    """)
    return


@app.cell
def _(cache_parquet, cluster_significant_differences_df, taxonomy_lf):
    from src.dataframes.significant_taxa_images import SignificantTaxaImagesSchema

    significant_taxa_images_df = cache_parquet(
        SignificantTaxaImagesSchema.build_df(
            cluster_significant_differences_df,
            taxonomy_lf.collect(engine="streaming"),
        ),
        cache_key=SignificantTaxaImagesSchema,
    ).collect(engine="streaming")
    return (significant_taxa_images_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Preview
    """)
    return


@app.cell
def _(significant_taxa_images_df):
    significant_taxa_images_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Write output for frontend
    """)
    return


@app.cell
def _(
    cluster_boundary_df,
    cluster_colors_df,
    cluster_significant_differences_df,
    significant_taxa_images_df,
    taxonomy_lf,
):
    from src.output import write_json_output

    taxonomy_df = taxonomy_lf.collect(engine="streaming")

    # write_json_output(
    #     cluster_significant_differences_df,
    #     cluster_boundary_df,
    #     taxonomy_df,
    #     cluster_colors_df,
    #     significant_taxa_images_df,
    #     "/dev/stdout",
    # )

    write_json_output(
        cluster_significant_differences_df,
        cluster_boundary_df,
        taxonomy_df,
        cluster_colors_df,
        significant_taxa_images_df,
        "frontend/aggregations.json",
    )
    return


if __name__ == "__main__":
    app.run()
