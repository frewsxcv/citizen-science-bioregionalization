# pyright: reportUnusedExpression=false

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

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


@app.cell
def _(mo):
    num_clusters_ui = mo.ui.number(value=10, label="Number of clusters")
    num_clusters_ui
    return (num_clusters_ui,)


@app.cell(hide_code=True)
def _(mo):
    taxon_filter_ui = mo.ui.text("", label="Taxon filter (optional)")
    taxon_filter_ui
    return (taxon_filter_ui,)


@app.cell(hide_code=True)
def _(mo):
    limit_results_enabled_ui = mo.ui.checkbox(value=True, label="Enable limit")
    return (limit_results_enabled_ui,)


@app.cell(hide_code=True)
def _(limit_results_enabled_ui, mo):
    limit_results_value_ui = mo.ui.number(
        value=1000, label="Limit results", disabled=not limit_results_enabled_ui.value
    )
    return (limit_results_value_ui,)


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
def _(mo):
    min_lon_ui = mo.ui.number(value=-87.0, label="Min Longitude")
    min_lat_ui = mo.ui.number(value=25.0, label="Min Latitude")
    mo.vstack([min_lon_ui, min_lat_ui])
    return min_lat_ui, min_lon_ui


@app.cell(hide_code=True)
def _(mo):
    max_lat_ui = mo.ui.number(value=47.0, label="Max Latitude")
    max_lon_ui = mo.ui.number(value=-66.0, label="Max Longitude")
    mo.vstack([max_lon_ui, max_lat_ui])
    return max_lat_ui, max_lon_ui


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
def _(
    geocode_precision_ui,
    limit_results_enabled_ui,
    limit_results_value_ui,
    max_lat_ui,
    max_lon_ui,
    min_lat_ui,
    min_lon_ui,
    num_clusters_ui,
    parquet_source_path_ui,
    taxon_filter_ui,
):
    from src.cli import parse_args_with_defaults

    args = parse_args_with_defaults(
        geocode_precision=geocode_precision_ui.value,
        num_clusters=num_clusters_ui.value,
        taxon_filter=taxon_filter_ui.value,
        min_lat=min_lat_ui.value,
        max_lat=max_lat_ui.value,
        min_lon=min_lon_ui.value,
        max_lon=max_lon_ui.value,
        limit_results=limit_results_value_ui.value
        if limit_results_enabled_ui.value
        else None,
        parquet_source_path=parquet_source_path_ui.value,
    )
    return (args,)


@app.cell
def _(mo):
    run_button_ui = mo.ui.run_button()
    run_button_ui
    return (run_button_ui,)


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
def _(Bbox, args, mo, run_button_ui):
    from src.darwin_core_utils import load_darwin_core_data
    from src.dataframes.darwin_core import DarwinCoreSchema

    darwin_core_raw_lf = load_darwin_core_data(
        source_path=args.parquet_source_path,
        min_lat=args.min_lat,
        max_lat=args.max_lat,
        min_lon=args.min_lon,
        max_lon=args.max_lon,
        limit_results=args.limit_results,
        taxon_filter=args.taxon_filter,
    )

    darwin_core_lf = DarwinCoreSchema.build_lf(
        darwin_core_raw_lf,
        bounding_box=Bbox.from_coordinates(
            args.min_lat, args.max_lat, args.min_lon, args.max_lon
        ),
        limit=None,  # Already limited by load_darwin_core_data
    )

    if mo.running_in_notebook() and not args.no_stop:
        print("STOPPING")
        mo.stop(not run_button_ui.value)
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
def _(Bbox, args, cache_parquet, darwin_core_lf):
    from src.dataframes.geocode import GeocodeNoEdgesSchema, GeocodeSchema

    geocode_lf_with_edges = cache_parquet(
        GeocodeSchema.build_df(
            darwin_core_lf,
            args.geocode_precision,
            bounding_box=Bbox.from_coordinates(
                args.min_lat, args.max_lat, args.min_lon, args.max_lon
            ),
        ),
        cache_key=GeocodeSchema,
    )

    geocode_lf = cache_parquet(
        GeocodeNoEdgesSchema.from_geocode_schema(
            geocode_lf_with_edges,
        ),
        cache_key=GeocodeNoEdgesSchema,
    )

    geocode_lf
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
def _(Bbox, args, cache_parquet, darwin_core_lf, geocode_lf):
    from src.dataframes.taxonomy import TaxonomySchema

    taxonomy_lf = cache_parquet(
        TaxonomySchema.build_lf(
            darwin_core_lf,
            args.geocode_precision,
            geocode_lf,
            bounding_box=Bbox.from_coordinates(
                args.min_lat, args.max_lat, args.min_lon, args.max_lon
            ),
        ),
        cache_key=TaxonomySchema,
    )

    taxonomy_lf
    return (taxonomy_lf,)


@app.cell
def _(taxonomy_lf):
    taxonomy_lf.limit(100).collect(engine="streaming")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## `GeocodeSpeciesCounts`
    """)
    return


@app.cell
def _(
    Bbox,
    args,
    cache_parquet,
    darwin_core_lf,
    geocode_lf,
    taxonomy_lf,
):
    from src.dataframes.geocode_taxa_counts import GeocodeTaxaCountsSchema

    geocode_taxa_counts_lf = cache_parquet(
        GeocodeTaxaCountsSchema.build_df(
            darwin_core_lf,
            args.geocode_precision,
            taxonomy_lf,
            geocode_lf,
            bounding_box=Bbox.from_coordinates(
                args.min_lat, args.max_lat, args.min_lon, args.max_lon
            ),
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
    args,
    cache_parquet,
    geocode_connectivity_matrix,
    geocode_distance_matrix,
    geocode_lf,
):
    from src.dataframes.geocode_cluster import GeocodeClusterSchema

    geocode_cluster_lf = cache_parquet(
        GeocodeClusterSchema.build_df(
            geocode_lf,
            geocode_distance_matrix,
            geocode_connectivity_matrix,
            args.num_clusters,
        ),
        cache_key=GeocodeClusterSchema,
    )
    return (geocode_cluster_lf,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Preview
    """)
    return


@app.cell
def _(geocode_cluster_lf):
    geocode_cluster_lf.limit(100).collect()
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
def _(cache_parquet, geocode_cluster_lf, geocode_lf):
    from src.dataframes.cluster_neighbors import ClusterNeighborsSchema

    cluster_neighbors_lf = cache_parquet(
        ClusterNeighborsSchema.build_df(
            geocode_lf,
            geocode_cluster_lf.collect(),
        ),
        cache_key=ClusterNeighborsSchema,
    )
    return (cluster_neighbors_lf,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Preview
    """)
    return


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Build
    """)
    return


@app.cell
def _(
    cache_parquet,
    geocode_cluster_lf,
    geocode_taxa_counts_lf,
    taxonomy_lf,
):
    from src.dataframes.cluster_taxa_statistics import ClusterTaxaStatisticsSchema

    cluster_taxa_statistics_df = cache_parquet(
        ClusterTaxaStatisticsSchema.build_df(
            geocode_taxa_counts_lf,
            geocode_cluster_lf,
            taxonomy_lf,
        ),
        cache_key=ClusterTaxaStatisticsSchema,
    ).collect(engine="streaming")
    return (cluster_taxa_statistics_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Preview
    """)
    return


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Build
    """)
    return


@app.cell
def _(
    cache_parquet,
    cluster_neighbors_lf,
    cluster_taxa_statistics_df,
):
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Preview
    """)
    return


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Build
    """)
    return


@app.cell
def _(cache_parquet, geocode_cluster_lf, geocode_lf):
    from src.dataframes.cluster_boundary import ClusterBoundarySchema

    cluster_boundary_df = cache_parquet(
        ClusterBoundarySchema.build_df(
            geocode_cluster_lf.collect(),
            geocode_lf,
        ),
        cache_key=ClusterBoundarySchema,
    ).collect(engine="streaming")
    return (cluster_boundary_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Preview
    """)
    return


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Build
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
    ## `ClusterColor`
    """)
    return


@app.cell
def _(
    cache_parquet,
    cluster_boundary_df,
    cluster_neighbors_lf,
    cluster_taxa_statistics_df,
):
    from src.dataframes.cluster_color import ClusterColorSchema

    cluster_colors_df = cache_parquet(
        ClusterColorSchema.build_df(
            cluster_neighbors_lf,
            cluster_boundary_df,
            cluster_taxa_statistics_df,
            color_method="taxonomic",
            # color_method="geographic",
        ),
        cache_key=ClusterColorSchema,
    ).collect(engine="streaming")

    cluster_colors_df
    return (cluster_colors_df,)


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
def _(
    cache_parquet,
    geocode_cluster_lf,
    geocode_distance_matrix,
    geocode_lf,
):
    from src.dataframes.permanova_results import PermanovaResultsSchema

    permanova_results_df = cache_parquet(
        PermanovaResultsSchema.build_df(
            geocode_distance_matrix=geocode_distance_matrix,
            geocode_cluster_df=geocode_cluster_lf.collect(),
            geocode_lf=geocode_lf,
        ),
        cache_key=PermanovaResultsSchema,
    ).collect(engine="streaming")
    return (permanova_results_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Preview
    """)
    return


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Build
    """)
    return


@app.cell
def _(cache_parquet, geocode_cluster_lf, geocode_distance_matrix):
    from src.dataframes.geocode_silhouette_score import GeocodeSilhouetteScoreSchema

    geocode_silhouette_score_df = cache_parquet(
        GeocodeSilhouetteScoreSchema.build_df(
            geocode_distance_matrix,
            geocode_cluster_lf.collect(),
        ),
        cache_key=GeocodeSilhouetteScoreSchema,
    ).collect(engine="streaming")
    return (geocode_silhouette_score_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Preview
    """)
    return


@app.cell
def _(geocode_silhouette_score_df):
    geocode_silhouette_score_df.sort(by="silhouette_score")
    return


@app.cell
def _(
    cluster_colors_df,
    geocode_cluster_lf,
    geocode_distance_matrix,
    geocode_silhouette_score_df,
):
    from src.plot.silhouette_score import plot_silhouette_scores

    plot_silhouette_scores(
        geocode_cluster_lf.collect(),
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
    cluster_colors_df,
    geocode_cluster_lf,
    geocode_distance_matrix,
):
    from src.plot.dimnesionality_reduction import create_dimensionality_reduction_plot

    create_dimensionality_reduction_plot(
        geocode_distance_matrix,
        geocode_cluster_lf.collect(),
        cluster_colors_df,
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
    cluster_colors_df,
    cluster_significant_differences_df,
    cluster_taxa_statistics_df,
    geocode_cluster_lf,
    geocode_distance_matrix,
    geocode_lf,
    geocode_taxa_counts_lf,
    mo,
    taxonomy_lf,
):
    from src.plot.cluster_taxa import create_cluster_taxa_heatmap

    heatmap = create_cluster_taxa_heatmap(
        geocode_lf=geocode_lf,
        geocode_cluster_df=geocode_cluster_lf.collect(),
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
def _(
    cache_parquet,
    cluster_significant_differences_df,
    taxonomy_lf,
):
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
