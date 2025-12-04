# pyright: reportUnusedExpression=false

import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    from pathlib import Path

    import folium
    import marimo as mo
    import numpy as np
    import polars as pl
    import polars_darwin_core
    return Path, folium, mo, np, pl, polars_darwin_core


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
    log_file_ui
    return


@app.cell(hide_code=True)
def _(mo):
    parquet_source_path_ui = mo.ui.text(
        "gs://public-datasets-gbif/occurrence/2025-11-01/occurrence.parquet/",
        label="Input GCS directory",
    )
    parquet_source_path_ui
    return (parquet_source_path_ui,)


@app.cell(hide_code=True)
def _(mo):
    geocode_precision_ui = mo.ui.number(value=4, label="Geocode precision")
    geocode_precision_ui
    return (geocode_precision_ui,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Cluster Count Selection

    Choose between manual selection or automatic optimization.
    """)
    return


@app.cell
def _(mo):
    use_auto_k_ui = mo.ui.checkbox(
        value=True, label="Use automatic cluster optimization"
    )
    use_auto_k_ui
    return (use_auto_k_ui,)


@app.cell
def _(mo, use_auto_k_ui):
    # Always create num_clusters_ui to avoid None reference errors
    # But only display it when not using auto optimization
    num_clusters_ui = mo.ui.number(value=10, label="Number of clusters (manual)")

    if not use_auto_k_ui.value:
        display_manual = num_clusters_ui
    else:
        display_manual = mo.md("*Using automatic optimization (see below)*")

    display_manual
    return (num_clusters_ui,)


@app.cell
def _(mo, use_auto_k_ui):
    # Automatic optimization parameters
    if use_auto_k_ui.value:
        k_min_ui = mo.ui.number(value=5, label="Minimum clusters to evaluate")
        k_max_ui = mo.ui.number(value=15, label="Maximum clusters to evaluate")
        optimization_method_ui = mo.ui.dropdown(
            options={
                "multi_criteria": "Multi-Criteria (Recommended)",
                "silhouette": "Best Silhouette Score",
                "elbow": "Elbow Method",
                "compromise": "Quality Compromise",
            },
            value="multi_criteria",
            label="Selection method",
        )
        display_auto = mo.vstack(
            [
                mo.hstack([k_min_ui, k_max_ui]),
                optimization_method_ui,
            ]
        )
    else:
        k_min_ui = None
        k_max_ui = None
        optimization_method_ui = None
        display_auto = mo.md("")

    display_auto
    return k_max_ui, k_min_ui, optimization_method_ui


@app.cell(hide_code=True)
def _(mo):
    taxon_filter_ui = mo.ui.text("", label="Taxon filter (optional)")
    taxon_filter_ui
    return (taxon_filter_ui,)


@app.cell(hide_code=True)
def _(mo):
    limit_results_ui = mo.ui.number(value=1000, label="Limit results")
    limit_results_ui
    return (limit_results_ui,)


@app.cell(hide_code=True)
def _(mo):
    min_lon_ui = mo.ui.number(value=-24.3261840479, label="Min Longitude")
    min_lat_ui = mo.ui.number(value=63.4963829617, label="Min Latitude")
    mo.vstack([min_lon_ui, min_lat_ui])
    return min_lat_ui, min_lon_ui


@app.cell(hide_code=True)
def _(mo):
    max_lat_ui = mo.ui.number(value=66.5267923041, label="Max Latitude")
    max_lon_ui = mo.ui.number(value=-13.609732225, label="Max Longitude")
    mo.vstack([max_lon_ui, max_lat_ui])
    return max_lat_ui, max_lon_ui


@app.cell(hide_code=True)
def _(
    geocode_precision_ui,
    limit_results_ui,
    max_lat_ui,
    max_lon_ui,
    min_lat_ui,
    min_lon_ui,
    num_clusters_ui,
    parquet_source_path_ui,
    taxon_filter_ui,
):
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
    parser.add_argument(
        "--min-lat",
        type=float,
        default=min_lat_ui.value,
        help="Minimum latitude for bounding box",
    )
    parser.add_argument(
        "--max-lat",
        type=float,
        default=max_lat_ui.value,
        help="Maximum latitude for bounding box",
    )
    parser.add_argument(
        "--min-lon",
        type=float,
        default=min_lon_ui.value,
        help="Minimum longitude for bounding box",
    )
    parser.add_argument(
        "--max-lon",
        type=float,
        default=max_lon_ui.value,
        help="Maximum longitude for bounding box",
    )
    parser.add_argument(
        "--limit-results",
        type=int,
        default=limit_results_ui.value,
        help="Limit the number of results fetched",
    )

    # Positional arguments
    parser.add_argument(
        "parquet_source_path",
        type=str,
        nargs="?",
        help="Path to the parquet data source",
        default=parquet_source_path_ui.value,
    )

    args = parser.parse_args()
    return (args,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Set up logging
    """)
    return


@app.cell
def _(args):
    import logging

    logging.basicConfig(filename=args.log_file, encoding="utf-8", level=logging.INFO)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## `DarwinCore`
    """)
    return


@app.cell
def _(Path, args, pl, polars_darwin_core):
    # Detect if source is a Darwin Core archive (directory with meta.xml) or parquet
    source_path = Path(args.parquet_source_path)
    is_darwin_core_archive = (
        source_path.is_dir() and (source_path / "meta.xml").exists()
    )

    # Build base filters for geographic bounds
    # Use camelCase column names (Darwin Core standard)
    # First filter out null coordinates, then apply bounds
    base_filters = (
        pl.col("decimalLatitude").is_not_null()
        & pl.col("decimalLongitude").is_not_null()
        & (pl.col("decimalLatitude") >= args.min_lat)
        & (pl.col("decimalLatitude") <= args.max_lat)
        & (pl.col("decimalLongitude") >= args.min_lon)
        & (pl.col("decimalLongitude") <= args.max_lon)
    )

    # Add taxon filter if specified
    if args.taxon_filter:
        taxon_filter_expr = (
            (pl.col("kingdom") == args.taxon_filter)
            | (pl.col("phylum") == args.taxon_filter)
            | (pl.col("class") == args.taxon_filter)
            | (pl.col("order") == args.taxon_filter)
            | (pl.col("family") == args.taxon_filter)
            | (pl.col("genus") == args.taxon_filter)
            | (pl.col("species") == args.taxon_filter)
        )
        base_filters = base_filters & taxon_filter_expr

    # Mapping from lowercase (parquet snapshots) to camelCase (Darwin Core standard)
    parquet_to_darwin_core_columns = {
        "decimallatitude": "decimalLatitude",
        "decimallongitude": "decimalLongitude",
        "taxonkey": "taxonKey",
        "specieskey": "speciesKey",
        "acceptedtaxonkey": "acceptedTaxonKey",
        "kingdomkey": "kingdomKey",
        "phylumkey": "phylumKey",
        "classkey": "classKey",
        "orderkey": "orderKey",
        "familykey": "familyKey",
        "genuskey": "genusKey",
        "subgenuskey": "subgenusKey",
        "taxonrank": "taxonRank",
        "scientificname": "scientificName",
        "verbatimscientificname": "verbatimScientificName",
        "countrycode": "countryCode",
        "gbifid": "gbifID",
        "datasetkey": "datasetKey",
        "occurrenceid": "occurrenceID",
        "eventdate": "eventDate",
        "basisofrecord": "basisOfRecord",
        "individualcount": "individualCount",
        "publishingorgkey": "publishingOrgKey",
        "coordinateuncertaintyinmeters": "coordinateUncertaintyInMeters",
        "coordinateprecision": "coordinatePrecision",
        "hascoordinate": "hasCoordinate",
        "hasgeospatialissues": "hasGeospatialIssues",
        "stateprovince": "stateProvince",
        "iucnredlistcategory": "iucnRedListCategory",
    }

    if is_darwin_core_archive:
        # Load from Darwin Core archive (already uses camelCase)
        darwin_core_lazy_frame = polars_darwin_core.DarwinCoreLazyFrame(
            polars_darwin_core.DarwinCoreLazyFrame.from_archive(
                args.parquet_source_path
            )
            ._inner.filter(base_filters)
            .limit(args.limit_results)
        )
    else:
        # Load from parquet snapshot and rename columns to camelCase
        # Public GCS buckets (like GBIF) are accessible without credentials
        inner_lf = pl.scan_parquet(args.parquet_source_path)
        # Only rename columns that actually exist in the parquet file
        existing_columns = set(inner_lf.collect_schema().names())
        columns_to_rename = {
            k: v
            for k, v in parquet_to_darwin_core_columns.items()
            if k in existing_columns
        }
        inner_lf = inner_lf.rename(columns_to_rename)
        darwin_core_lazy_frame = polars_darwin_core.DarwinCoreLazyFrame(
            inner_lf.filter(base_filters).limit(args.limit_results)
        )
    return (darwin_core_lazy_frame,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## `Geocode`
    """)
    return


@app.cell
def _(args, darwin_core_lazy_frame):
    from src.dataframes.geocode import GeocodeNoEdgesSchema, GeocodeSchema
    from src.types import Bbox

    geocode_dataframe_with_edges = GeocodeSchema.build(
        darwin_core_lazy_frame,
        args.geocode_precision,
        bounding_box=Bbox.from_coordinates(
            args.min_lat, args.max_lat, args.min_lon, args.max_lon
        ),
    )

    geocode_dataframe = GeocodeNoEdgesSchema.from_geocode_schema(
        geocode_dataframe_with_edges,
    )

    geocode_dataframe
    return geocode_dataframe, geocode_dataframe_with_edges


@app.cell(hide_code=True)
def _(folium, geocode_dataframe, geocode_dataframe_with_edges, pl):
    _center = geocode_dataframe.select(
        pl.col("center").alias("geometry"),
    )
    _boundary = geocode_dataframe_with_edges.select(
        pl.col("boundary").alias("geometry"),
        pl.col("is_edge"),
    )

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
def _(darwin_core_lazy_frame):
    from src.dataframes.taxonomy import TaxonomySchema

    taxonomy_dataframe = TaxonomySchema.build(darwin_core_lazy_frame)

    taxonomy_dataframe
    return (taxonomy_dataframe,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## `GeocodeSpeciesCounts`
    """)
    return


@app.cell
def _(args, darwin_core_lazy_frame, geocode_dataframe, taxonomy_dataframe):
    from src.dataframes.geocode_taxa_counts import GeocodeTaxaCountsSchema

    geocode_taxa_counts_dataframe = GeocodeTaxaCountsSchema.build(
        darwin_core_lazy_frame,
        args.geocode_precision,
        taxonomy_dataframe,
        geocode_dataframe,
    )

    geocode_taxa_counts_dataframe
    return (geocode_taxa_counts_dataframe,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## `GeocodeConnectivity`
    """)
    return


@app.cell
def _(geocode_dataframe):
    from src.matrices.geocode_connectivity import GeocodeConnectivityMatrix

    geocode_connectivity_matrix = GeocodeConnectivityMatrix.build(geocode_dataframe)

    geocode_connectivity_matrix._connectivity_matrix
    return (geocode_connectivity_matrix,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## `GeocodeDistance`
    """)
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
    mo.md(r"""
    ## Cluster Optimization (Optional)
    """)
    return


@app.cell(hide_code=True)
def _(mo, use_auto_k_ui):
    if use_auto_k_ui.value:
        display_optimization_header = mo.md(r"""
        ### Running Automatic Cluster Selection

        Evaluating multiple cluster counts using statistical metrics...
        """)
    else:
        display_optimization_header = mo.md("")

    display_optimization_header
    return


@app.cell
def _(
    geocode_connectivity_matrix,
    geocode_dataframe,
    geocode_distance_matrix,
    k_max_ui,
    k_min_ui,
    optimization_method_ui,
    use_auto_k_ui,
):
    cluster_optimization_result = None
    cluster_metrics_list = None

    if use_auto_k_ui.value:
        from src.cluster_optimization import ClusterOptimizer

        optimizer = ClusterOptimizer(
            geocode_dataframe=geocode_dataframe,
            distance_matrix=geocode_distance_matrix,
            connectivity_matrix=geocode_connectivity_matrix,
        )

        cluster_metrics_list = optimizer.evaluate_k_range(
            k_min=k_min_ui.value,
            k_max=k_max_ui.value,
        )

        cluster_optimization_result = optimizer.suggest_optimal_k(
            cluster_metrics_list, method=optimization_method_ui.value
        )
    return (cluster_optimization_result,)


@app.cell(hide_code=True)
def _(cluster_optimization_result, mo, use_auto_k_ui):
    if use_auto_k_ui.value and cluster_optimization_result:
        from src.cluster_optimization import create_metrics_report

        report = create_metrics_report(cluster_optimization_result)
        display_report = mo.md(f"```\n{report}\n```")
    else:
        display_report = mo.md("")

    display_report
    return


@app.cell(hide_code=True)
def _(cluster_optimization_result, mo, use_auto_k_ui):
    if use_auto_k_ui.value and cluster_optimization_result:
        from src.plot.cluster_optimization import plot_optimization_metrics

        fig = plot_optimization_metrics(cluster_optimization_result)
        display_optimization_plot = mo.mpl.interactive(fig)
    else:
        display_optimization_plot = mo.md("")

    display_optimization_plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## `GeocodeCluster`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Build
    """)
    return


@app.cell
def _(
    cluster_optimization_result,
    geocode_connectivity_matrix,
    geocode_dataframe,
    geocode_distance_matrix,
    num_clusters_ui,
    use_auto_k_ui,
):
    from src.dataframes.geocode_cluster import GeocodeClusterSchema

    # Determine which k to use
    if use_auto_k_ui.value and cluster_optimization_result:
        optimal_k = cluster_optimization_result.optimal_k
    else:
        optimal_k = num_clusters_ui.value

    geocode_cluster_dataframe = GeocodeClusterSchema.build(
        geocode_dataframe,
        geocode_distance_matrix,
        geocode_connectivity_matrix,
        optimal_k,
    )
    return geocode_cluster_dataframe, optimal_k


@app.cell(hide_code=True)
def _(mo, optimal_k, use_auto_k_ui):
    if use_auto_k_ui.value:
        display_k_used = mo.md(
            f"**Using k={optimal_k} clusters (automatically selected)**"
        )
    else:
        display_k_used = mo.md(f"**Using k={optimal_k} clusters (manually specified)**")

    display_k_used
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Preview
    """)
    return


@app.cell
def _(geocode_cluster_dataframe):
    geocode_cluster_dataframe.limit(3)
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
    mo.md(r"""
    ## `ClusterNeighbors`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Build
    """)
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
    mo.md(r"""
    ### Preview
    """)
    return


@app.cell
def _(cluster_neighbors_dataframe):
    cluster_neighbors_dataframe
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## `ClusterTaxaStatistics`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Build
    """)
    return


@app.cell
def _(
    geocode_cluster_dataframe,
    geocode_taxa_counts_dataframe,
    taxonomy_dataframe,
):
    from src.dataframes.cluster_taxa_statistics import ClusterTaxaStatisticsSchema

    cluster_taxa_statistics_dataframe = ClusterTaxaStatisticsSchema.build(
        geocode_taxa_counts_dataframe,
        geocode_cluster_dataframe,
        taxonomy_dataframe,
    )
    return (cluster_taxa_statistics_dataframe,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Preview
    """)
    return


@app.cell
def _(cluster_taxa_statistics_dataframe):
    cluster_taxa_statistics_dataframe
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## `ClusterSignificantDifferences`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Build
    """)
    return


@app.cell
def _(cluster_neighbors_dataframe, cluster_taxa_statistics_dataframe):
    from src.dataframes.cluster_significant_differences import (
        ClusterSignificantDifferencesSchema,
    )

    cluster_significant_differences_dataframe = (
        ClusterSignificantDifferencesSchema.build(
            cluster_taxa_statistics_dataframe,
            cluster_neighbors_dataframe,
        )
    )
    return (cluster_significant_differences_dataframe,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Preview
    """)
    return


@app.cell
def _(cluster_significant_differences_dataframe):
    cluster_significant_differences_dataframe
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## `ClusterBoundary`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Build
    """)
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
    mo.md(r"""
    ### Preview
    """)
    return


@app.cell
def _(cluster_boundary_dataframe):
    cluster_boundary_dataframe
    return


@app.cell(hide_code=True)
def _(cluster_boundary_dataframe, folium):
    _boundary = cluster_boundary_dataframe.select(["geometry", "cluster"])

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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Build
    """)
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
    mo.md(r"""
    ### Preview
    """)
    return


@app.cell
def _(cluster_distance_matrix):
    cluster_distance_matrix.squareform()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## `ClusterColors`

    **Note:** Taxonomic coloring requires ≥10 clusters for UMAP dimensionality reduction.
    If fewer clusters exist, the system automatically falls back to geographic coloring.
    """)
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
    mo.md(r"""
    ## `PermanovaResults`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Build
    """)
    return


@app.cell
def _(geocode_cluster_dataframe, geocode_dataframe, geocode_distance_matrix):
    from src.dataframes.permanova_results import PermanovaResultsSchema

    permanova_results_dataframe = PermanovaResultsSchema.build(
        geocode_distance_matrix=geocode_distance_matrix,
        geocode_cluster_dataframe=geocode_cluster_dataframe,
        geocode_dataframe=geocode_dataframe,
    )
    return (permanova_results_dataframe,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Preview
    """)
    return


@app.cell
def _(permanova_results_dataframe):
    permanova_results_dataframe
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## `GeocodeSilhouetteScore`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Build
    """)
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
    mo.md(r"""
    ### Preview
    """)
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
    mo.md(r"""
    ## Build and plot GeoJSON feature collection
    """)
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
    mo.md(r"""
    ## Dimensionality reduction plot
    """)
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
    mo.md(r"""
    ## Clustermap visualization
    """)
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
    mo,
    taxonomy_dataframe,
):
    from src.plot.cluster_taxa import create_cluster_taxa_heatmap

    heatmap = create_cluster_taxa_heatmap(
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
def _(cluster_significant_differences_dataframe, taxonomy_dataframe):
    from src.dataframes.significant_taxa_images import SignificantTaxaImagesSchema

    significant_taxa_images_dataframe = SignificantTaxaImagesSchema.build(
        cluster_significant_differences_dataframe,
        taxonomy_dataframe,
    )
    return (significant_taxa_images_dataframe,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Preview
    """)
    return


@app.cell
def _(significant_taxa_images_dataframe):
    significant_taxa_images_dataframe
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Write output for frontend
    """)
    return


@app.cell
def _(
    cluster_boundary_dataframe,
    cluster_colors_dataframe,
    cluster_significant_differences_dataframe,
    significant_taxa_images_dataframe,
    taxonomy_dataframe,
):
    from src.json_output import write_json_output

    # write_json_output(
    #     cluster_significant_differences_dataframe,
    #     cluster_boundary_dataframe,
    #     taxonomy_dataframe,
    #     cluster_colors_dataframe,
    #     significant_taxa_images_dataframe,
    #     "/dev/stdout",
    # )

    write_json_output(
        cluster_significant_differences_dataframe,
        cluster_boundary_dataframe,
        taxonomy_dataframe,
        cluster_colors_dataframe,
        significant_taxa_images_dataframe,
        "frontend/aggregations.json",
    )
    return


if __name__ == "__main__":
    app.run()
