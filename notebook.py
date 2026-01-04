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

    return cache_parquet, folium, mo, np, pl


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
    # Define Marimo input UI elements

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
    max_taxa_enabled_ui = mo.ui.checkbox(value=False, label="Limit to top N taxa")
    max_taxa_value_ui = mo.ui.number(
        value=5000, label="Keep top N taxa by occurrence count"
    )
    min_geocode_presence_enabled_ui = mo.ui.checkbox(
        value=False, label="Filter rare taxa"
    )
    min_geocode_presence_value_ui = mo.ui.number(
        value=0.05, label="Min fraction of hexagons a taxon must appear in", step=0.01
    )
    min_lon_ui = mo.ui.number(value=-87.0, label="Longitude")
    min_lat_ui = mo.ui.number(value=25.0, label="Latitude")
    max_lon_ui = mo.ui.number(value=-66.0, label="Longitude")
    max_lat_ui = mo.ui.number(value=47.0, label="Latitude")
    run_button_ui = mo.ui.run_button()
    return (
        geocode_precision_ui,
        limit_results_enabled_ui,
        limit_results_value_ui,
        log_file_ui,
        max_clusters_to_test_ui,
        max_lat_ui,
        max_lon_ui,
        max_taxa_enabled_ui,
        max_taxa_value_ui,
        min_clusters_to_test_ui,
        min_geocode_presence_enabled_ui,
        min_geocode_presence_value_ui,
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
def _(max_clusters_to_test_ui, min_clusters_to_test_ui, mo):
    _description = mo.md(
        "**Cluster Configuration:** Number of clusters will be automatically optimized based on silhouette scores."
    )

    mo.vstack(
        [
            _description,
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
def _(
    max_taxa_enabled_ui,
    max_taxa_value_ui,
    min_geocode_presence_enabled_ui,
    min_geocode_presence_value_ui,
    mo,
):
    _description = mo.md(
        "**Taxa Filtering:** Reduce dimensionality before pivoting by filtering to most informative taxa. "
        "This significantly speeds up distance matrix computation for large datasets."
    )

    _max_taxa_description = mo.md(
        "_Keep only the N most abundant taxa (by total occurrence count). "
        "Recommended: 5,000–10,000 for large datasets._"
    )

    _min_presence_description = mo.md(
        "_Remove rare taxa that appear in too few hexagons. "
        "Taxa seen in very few locations add noise but don't help distinguish bioregions. "
        "For example, 0.05 means a taxon must appear in at least 5% of hexagons to be included. "
        "Recommended: 0.02–0.05 (2–5%)._"
    )

    mo.vstack(
        [
            _description,
            mo.hstack([max_taxa_enabled_ui, max_taxa_value_ui]),
            _max_taxa_description,
            mo.hstack([min_geocode_presence_enabled_ui, min_geocode_presence_value_ui]),
            _min_presence_description,
        ]
    )
    return


@app.cell(hide_code=True)
def _(folium, max_lat_ui, max_lon_ui, min_lat_ui, min_lon_ui, mo):
    def build_map():
        map = folium.Map(
            tiles="Esri.WorldGrayCanvas",
        )

        bounds = [
            [min_lat_ui.value, min_lon_ui.value],
            [max_lat_ui.value, max_lon_ui.value],
        ]

        folium.Rectangle(bounds=bounds).add_to(map)

        map.fit_bounds(bounds, padding=[20, 20])

        return map

    mo.hstack(
        [
            mo.vstack(
                [
                    mo.md("**Minimum**"),
                    min_lat_ui,
                    min_lon_ui,
                    mo.md("**Maximum**"),
                    max_lat_ui,
                    max_lon_ui,
                ]
            ),
            build_map(),
        ],
        widths="equal",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Resolved Inputs
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
    max_taxa_enabled_ui,
    max_taxa_value_ui,
    min_clusters_to_test_ui,
    min_geocode_presence_enabled_ui,
    min_geocode_presence_value_ui,
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
        max_taxa=max_taxa_value_ui.value if max_taxa_enabled_ui.value else None,
        min_geocode_presence=min_geocode_presence_value_ui.value
        if min_geocode_presence_enabled_ui.value
        else None,
    )
    return (args,)


@app.cell(hide_code=True)
def _(args, mo, run_button_ui):
    from src.types import Bbox

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
    max_taxa = args.max_taxa
    min_geocode_presence = args.min_geocode_presence
    bounding_box = Bbox.from_coordinates(min_lat, max_lat, min_lon, max_lon)

    inputs_table = mo.ui.table(
        label="Inputs",
        selection=None,
        pagination=False,
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
            {"variable": "max_taxa", "value": max_taxa},
            {"variable": "min_geocode_presence", "value": min_geocode_presence},
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
        bounding_box,
        geocode_precision,
        limit_results,
        log_file,
        max_clusters_to_test,
        max_taxa,
        min_clusters_to_test,
        min_geocode_presence,
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
    # Step 1. Fetch data
    """)
    return


@app.cell
def _(bounding_box, limit_results, parquet_source_path, taxon_filter):
    from src.dataframes.darwin_core import build_darwin_core_lf

    darwin_core_lf = build_darwin_core_lf(
        source_path=parquet_source_path,
        bounding_box=bounding_box,
        limit=limit_results,
        taxon_filter=taxon_filter,
    )
    return (darwin_core_lf,)


@app.cell
def _(darwin_core_lf, pl):
    darwin_core_lf.select(pl.len()).collect(engine="streaming")
    return


@app.cell
def _(bounding_box, cache_parquet, darwin_core_lf, geocode_precision):
    from src.dataframes.geocode import (
        GeocodeSchema,
        build_geocode_df,
    )

    geocode_lf_with_edges = cache_parquet(
        build_geocode_df(
            darwin_core_lf,
            geocode_precision,
            bounding_box=bounding_box,
        ),
        cache_key=GeocodeSchema,
    )
    return (geocode_lf_with_edges,)


@app.cell
def _(cache_parquet, geocode_lf_with_edges):
    from src.dataframes.geocode import (
        GeocodeNoEdgesSchema,
        build_geocode_no_edges_lf,
    )

    geocode_unfiltered_lf = cache_parquet(
        build_geocode_no_edges_lf(
            geocode_lf_with_edges,
        ),
        cache_key=GeocodeNoEdgesSchema,
    )
    return GeocodeNoEdgesSchema, geocode_unfiltered_lf


@app.cell
def _(geocode_lf_with_edges):
    from src.dataframes.geocode_neighbors import (
        build_geocode_neighbors_df,
    )

    # Build neighbors for all geocodes (including edges)
    geocode_neighbors_with_edges_df = build_geocode_neighbors_df(
        geocode_lf_with_edges.collect(),
    )
    return (geocode_neighbors_with_edges_df,)


@app.cell
def _(cache_parquet, geocode_lf, geocode_neighbors_with_edges_df):
    from src.dataframes.geocode_neighbors import (
        GeocodeNeighborsSchema,
        build_geocode_neighbors_no_edges_df,
    )

    # Build neighbors for filtered geocodes only
    geocode_neighbors_df = cache_parquet(
        build_geocode_neighbors_no_edges_df(
            geocode_neighbors_with_edges_df,
            geocode_lf.collect(),
        ),
        cache_key=GeocodeNeighborsSchema,
    ).collect()
    return (geocode_neighbors_df,)


@app.cell(hide_code=True)
def _(folium, geocode_lf_with_edges, geocode_unfiltered_lf, pl):
    _center = geocode_unfiltered_lf.select(
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


@app.cell
def _(
    bounding_box,
    cache_parquet,
    darwin_core_lf,
    geocode_precision,
    geocode_unfiltered_lf,
):
    from src.dataframes.taxonomy import TaxonomySchema, build_taxonomy_lf

    taxonomy_lf = cache_parquet(
        build_taxonomy_lf(
            darwin_core_lf,
            geocode_precision,
            geocode_unfiltered_lf,
            bounding_box=bounding_box,
        ),
        cache_key=TaxonomySchema,
    )
    return (taxonomy_lf,)


@app.cell
def _(taxonomy_lf):
    taxonomy_lf.limit(100).collect(engine="streaming")
    return


@app.cell
def _(
    bounding_box,
    cache_parquet,
    darwin_core_lf,
    geocode_precision,
    geocode_unfiltered_lf,
    taxonomy_lf,
):
    from src.dataframes.geocode_taxa_counts import (
        GeocodeTaxaCountsSchema,
        build_geocode_taxa_counts_lf,
    )

    geocode_taxa_counts_unfiltered_lf = cache_parquet(
        build_geocode_taxa_counts_lf(
            darwin_core_lf,
            geocode_precision,
            taxonomy_lf,
            geocode_unfiltered_lf,
            bounding_box=bounding_box,
        ),
        cache_key=GeocodeTaxaCountsSchema,
    )
    return (geocode_taxa_counts_unfiltered_lf,)


@app.cell
def _(geocode_taxa_counts_unfiltered_lf, max_taxa, min_geocode_presence):
    from src.dataframes.geocode_taxa_counts import (
        filter_top_taxa_lf,
    )

    # Apply taxa filtering if configured
    geocode_taxa_counts_lf = filter_top_taxa_lf(
        geocode_taxa_counts_unfiltered_lf,
        max_taxa=max_taxa,
        min_geocode_presence=min_geocode_presence,
    )
    return (geocode_taxa_counts_lf,)


@app.cell
def _(geocode_taxa_counts_lf):
    geocode_taxa_counts_lf.limit(100).collect(engine="streaming")
    return


@app.cell
def _(GeocodeNoEdgesSchema, geocode_taxa_counts_lf, geocode_unfiltered_lf, pl):
    # Filter geocode_unfiltered_lf to only include geocodes present in geocode_taxa_counts_lf
    geocode_lf = GeocodeNoEdgesSchema.validate(
        geocode_unfiltered_lf.join(
            geocode_taxa_counts_lf.select(pl.col("geocode").unique()),
            on="geocode",
            how="semi",
        ),
        eager=False,
    )
    return (geocode_lf,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Step 2. Cluster
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## `GeocodeConnectivity`
    """)
    return


@app.cell
def _(geocode_neighbors_df):
    from src.matrices.geocode_connectivity import GeocodeConnectivityMatrix

    geocode_connectivity_matrix = GeocodeConnectivityMatrix.build(geocode_neighbors_df)

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
    from src.dataframes.geocode_cluster import (
        GeocodeClusterMultiKSchema,
        build_geocode_cluster_multi_k_df,
    )

    all_clusters_df = cache_parquet(
        build_geocode_cluster_multi_k_df(
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
def _(all_clusters_df, geocode_distance_matrix):
    from src.cluster_optimization import optimize_num_clusters

    optimal_num_clusters, all_silhouette_scores = optimize_num_clusters(
        geocode_distance_matrix,
        all_clusters_df,
    )
    return all_silhouette_scores, optimal_num_clusters


@app.cell
def _(all_silhouette_scores, cache_parquet):
    # Cache the results
    from src.cluster_optimization import optimize_num_clusters as optimization_cache_key

    all_silhouette_scores_df = cache_parquet(
        all_silhouette_scores,
        cache_key=optimization_cache_key,
    ).collect(engine="streaming")
    return (all_silhouette_scores_df,)


@app.cell
def _(all_clusters_df, cache_parquet, optimal_num_clusters):
    # Create base GeocodeClusterSchema (single k) for downstream use
    from src.dataframes.geocode_cluster import (
        GeocodeClusterSchema,
        build_geocode_cluster_df,
    )

    geocode_cluster_df = cache_parquet(
        build_geocode_cluster_df(
            all_clusters_df,
            optimal_num_clusters,
        ),
        cache_key=GeocodeClusterSchema,
    ).collect(engine="streaming")
    return (geocode_cluster_df,)


@app.cell
def _(mo, optimal_num_clusters):
    mo.md(f"""
    **Optimal number of clusters: k={optimal_num_clusters}**
    """)
    return


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
def _(cache_parquet, geocode_cluster_df, geocode_neighbors_df):
    from src.dataframes.cluster_neighbors import (
        ClusterNeighborsSchema,
        build_cluster_neighbors_df,
    )

    cluster_neighbors_lf = cache_parquet(
        build_cluster_neighbors_df(
            geocode_neighbors_df,
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
    from src.dataframes.cluster_taxa_statistics import (
        ClusterTaxaStatisticsSchema,
        build_cluster_taxa_statistics_df,
    )

    cluster_taxa_statistics_df = cache_parquet(
        build_cluster_taxa_statistics_df(
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
        build_cluster_significant_differences_df,
    )

    cluster_significant_differences_df = cache_parquet(
        build_cluster_significant_differences_df(
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
    from src.dataframes.cluster_boundary import (
        ClusterBoundarySchema,
        build_cluster_boundary_df,
    )

    cluster_boundary_df = cache_parquet(
        build_cluster_boundary_df(
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
def _(optimal_num_clusters):
    # Use taxonomic coloring if we have at least 10 clusters, otherwise use geographic
    color_method = "taxonomic" if optimal_num_clusters >= 10 else "geographic"
    return (color_method,)


@app.cell
def _(
    cache_parquet,
    cluster_boundary_df,
    cluster_neighbors_lf,
    cluster_taxa_statistics_df,
    color_method,
):
    from src.dataframes.cluster_color import ClusterColorSchema, build_cluster_color_df

    cluster_colors_df = cache_parquet(
        build_cluster_color_df(
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
    # Step 3. Analyze
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## `PermanovaResults`
    """)
    return


@app.cell
def _(cache_parquet, geocode_cluster_df, geocode_distance_matrix, geocode_lf):
    from src.dataframes.permanova_results import (
        PermanovaResultsSchema,
        build_permanova_results_df,
    )

    permanova_results_df = cache_parquet(
        build_permanova_results_df(
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
    # Step 4. Export
    """)
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
    # from src.plot.cluster_taxa import create_cluster_taxa_heatmap

    # heatmap = create_cluster_taxa_heatmap(
    #     geocode_lf=geocode_lf,
    #     geocode_cluster_df=geocode_cluster_df,
    #     cluster_colors_df=cluster_colors_df,
    #     geocode_distance_matrix=geocode_distance_matrix,
    #     cluster_significant_differences_df=cluster_significant_differences_df,
    #     taxonomy_df=taxonomy_lf.collect(engine="streaming"),
    #     geocode_taxa_counts_lf=geocode_taxa_counts_lf,
    #     cluster_taxa_statistics_df=cluster_taxa_statistics_df,
    #     limit_species=5,
    # )

    # if heatmap is None:
    #     result = mo.md("No significant differences found between clusters.")
    # else:
    #     result = heatmap.figure

    # result
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## `SignificantTaxaImages`
    """)
    return


@app.cell
def _(cache_parquet, cluster_significant_differences_df, taxonomy_lf):
    from src.dataframes.significant_taxa_images import (
        SignificantTaxaImagesSchema,
        build_significant_taxa_images_df,
    )

    significant_taxa_images_df = cache_parquet(
        build_significant_taxa_images_df(
            cluster_significant_differences_df,
            taxonomy_lf.collect(engine="streaming"),
        ),
        cache_key=SignificantTaxaImagesSchema,
    ).collect(engine="streaming")
    return (significant_taxa_images_df,)


@app.cell(hide_code=True)
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
