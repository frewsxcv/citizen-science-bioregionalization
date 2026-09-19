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

    from src import defaults
    from src.materialize_parquet import materialize_parquet

    return materialize_parquet, defaults, folium, mo, np, pl


@app.cell
def _(mo):
    # Get CLI args (available when running with: marimo run notebook.py -- --arg=value)
    cli_args = mo.cli_args()
    return (cli_args,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🌿 Citizen Science Bioregionalization

    This notebook processes species occurrence records from citizen science datasets (e.g. GBIF Darwin Core Parquet sources), aggregates observations into H3 hexagonal geocodes, computes ecological dissimilarity matrices (e.g., Simpson turnover $\beta_{\text{sim}}$), and clusters geographic regions using spatially-constrained agglomerative hierarchical clustering.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Define inputs
    """)
    return


@app.cell(hide_code=True)
def _(cli_args, defaults, mo):
    # Define Marimo input UI elements
    # CLI args override defaults when provided (e.g., marimo run notebook.py -- --geocode-precision=5)

    log_file_ui = mo.ui.text(
        cli_args.get("log-file", defaults.LOG_FILE), label="Log file"
    )
    parquet_source_path_ui = mo.ui.text(
        cli_args.get("parquet-source-path", defaults.PARQUET_SOURCE_PATH),
        label="Input GCS directory",
    )
    geocode_precision_ui = mo.ui.number(
        value=cli_args.get("geocode-precision", defaults.GEOCODE_PRECISION),
        label="Geocode precision",
    )
    min_clusters_to_test_ui = mo.ui.number(
        value=cli_args.get("min-clusters", defaults.MIN_CLUSTERS),
        label="Min clusters to test",
    )
    max_clusters_to_test_ui = mo.ui.number(
        value=cli_args.get("max-clusters", defaults.MAX_CLUSTERS),
        label="Max clusters to test",
    )

    taxon_scope_ui = mo.ui.text(
        cli_args.get("scope", defaults.TAXON_SCOPE),
        label="Taxonomic scope (optional), e.g. order:Coleoptera",
    )
    # --no-limit is the only way to ask for a full run from the command line:
    # --limit-results can change how large the cap is but never switch it off,
    # since the default has it enabled.
    limit_results_enabled_ui = mo.ui.checkbox(
        value="no-limit" not in cli_args
        and ("limit-results" in cli_args or defaults.LIMIT_RESULTS_ENABLED),
        label="Enable limit",
    )
    limit_results_value_ui = mo.ui.number(
        value=cli_args.get("limit-results", defaults.LIMIT_RESULTS or 1000),
        label="Limit results",
    )
    # Distinct from --limit-results, which takes the head of the scan. At
    # snapshot scale the head is a sample of whichever source datasets sort
    # earliest, so a run that must cap its input for memory should cap it this
    # way instead. The head is still what the 1000-record interactive default
    # wants: it is there to make a run quick, and a uniform sample costs a
    # counting pass over the whole source to draw.
    sample_records_ui = mo.ui.number(
        value=cli_args.get("sample-records", defaults.SAMPLE_RECORDS or 0),
        label="Sample N records uniformly (0 = off)",
    )
    max_taxa_enabled_ui = mo.ui.checkbox(
        value="max-taxa" in cli_args or defaults.MAX_TAXA_ENABLED,
        label="Limit to top N taxa",
    )
    max_taxa_value_ui = mo.ui.number(
        value=cli_args.get("max-taxa", defaults.MAX_TAXA or 5000),
        label="Keep top N taxa by occurrence count",
    )
    min_geocode_presence_enabled_ui = mo.ui.checkbox(
        value="min-geocode-presence" in cli_args
        or defaults.MIN_GEOCODE_PRESENCE_ENABLED,
        label="Filter rare taxa",
    )
    min_geocode_presence_value_ui = mo.ui.number(
        value=cli_args.get(
            "min-geocode-presence", defaults.MIN_GEOCODE_PRESENCE or 0.05
        ),
        label="Min fraction of hexagons a taxon must appear in",
        step=0.01,
    )
    min_lon_ui = mo.ui.number(
        value=cli_args.get("min-lon", defaults.MIN_LON), label="Longitude"
    )
    min_lat_ui = mo.ui.number(
        value=cli_args.get("min-lat", defaults.MIN_LAT), label="Latitude"
    )
    max_lon_ui = mo.ui.number(
        value=cli_args.get("max-lon", defaults.MAX_LON), label="Longitude"
    )
    max_lat_ui = mo.ui.number(
        value=cli_args.get("max-lat", defaults.MAX_LAT), label="Latitude"
    )
    min_hex_records_ui = mo.ui.number(
        value=cli_args.get("min-hex-records", defaults.MIN_HEX_RECORDS or 0),
        label="Minimum records per hexagon (0 derives one from the data)",
    )
    # Opting out entirely, as distinct from pinning a value.
    no_hex_floor = "no-hex-floor" in cli_args
    # The findings page re-clusters each clade on its own, which is the run's
    # heaviest stage repeated twice over subsets. Worth it -- it is the only way
    # the congruence result gets recomputed rather than transcribed -- but a
    # run that just wants the map can skip it.
    no_findings = "no-findings" in cli_args
    # Where the findings page goes. Defaults into the gitignored output
    # directory; CI points it at whichever directory it uploads as the Pages
    # artifact, which differs per matrix entry.
    findings_output = str(
        cli_args.get("findings-output", defaults.FINDINGS_OUTPUT_PATH)
    )
    # Comma-separated silhouette,calinski_harabasz,davies_bouldin. Raising the
    # silhouette share raises the k chosen, because it is the only one of the
    # three with an interior optimum; the other two are monotone in k on real
    # data and simply vote for the end of the range.
    _raw_weights = cli_args.get("metric-weights")
    if _raw_weights:
        _parts = [p.strip() for p in str(_raw_weights).split(",")]
        if len(_parts) != 3:
            raise ValueError(
                f"--metric-weights={_raw_weights!r} needs three comma-separated "
                f"numbers: silhouette,calinski_harabasz,davies_bouldin."
            )
        try:
            _values = [float(p) for p in _parts]
        except ValueError as exc:
            raise ValueError(
                f"--metric-weights={_raw_weights!r} is not three numbers."
            ) from exc
        if any(v < 0 for v in _values) or sum(_values) <= 0:
            raise ValueError(
                f"--metric-weights={_raw_weights!r} must be non-negative and "
                f"not all zero."
            )
        metric_weights = dict(
            zip(("silhouette", "calinski_harabasz", "davies_bouldin"), _values)
        )
    else:
        metric_weights = dict(defaults.METRIC_WEIGHTS)

    # Ask for a specific number of regions. No metric here favours more of
    # them, so a larger k has to be requested rather than discovered.
    _pinned = cli_args.get("num-clusters")
    num_clusters_pinned = int(_pinned) if _pinned not in (None, "") else None

    composition_metric = cli_args.get(
        "composition-metric", defaults.COMPOSITION_METRIC
    )
    if composition_metric not in ("presence", "abundance", "betasim"):
        raise ValueError(
            f"Unknown --composition-metric={composition_metric!r}. "
            f"Expected 'presence', 'abundance' or 'betasim'."
        )
    linkage = cli_args.get("linkage", defaults.LINKAGE)
    if linkage not in ("average", "ward"):
        raise ValueError(
            f"Unknown --linkage={linkage!r}. Expected 'average' or 'ward'."
        )

    _levels = cli_args.get("hierarchy-levels", defaults.HIERARCHY_LEVELS)
    hierarchy_levels = (
        [int(x) for x in str(_levels).split(",") if x.strip()]
        if _levels not in (None, "")
        else None
    )
    # Which emitted cut the frontend opens on. Not the selector's k; see
    # defaults.DEFAULT_DISPLAY_LEVEL.
    default_display_level = int(
        cli_args.get("default-level", defaults.DEFAULT_DISPLAY_LEVEL)
    )
    reduction = cli_args.get("reduction", defaults.REDUCTION)
    if reduction not in ("umap", "pcoa"):
        raise ValueError(
            f"Unknown --reduction={reduction!r}. Expected 'umap' or 'pcoa'."
        )
    seed_ui = mo.ui.number(
        value=cli_args.get("seed", defaults.RANDOM_SEED if defaults.RANDOM_SEED is not None else 0),
        label="Random seed",
    )
    # For boolean flags, presence of the key means True (--no-stop becomes {'no-stop': ''})
    no_stop = "no-stop" in cli_args
    no_seed = "no-seed" in cli_args
    # Wikidata image lookup is the only network call after data loading; skipping
    # it keeps a run entirely offline.
    no_images = "no-images" in cli_args
    # Country-code filtering includes the maritime zone, so coastal clusters can
    # be driven by fish and seabirds rather than terrestrial biota.
    terrestrial_only = "terrestrial-only" in cli_args
    run_button_ui = mo.ui.run_button()
    return (
        geocode_precision_ui,
        limit_results_enabled_ui,
        limit_results_value_ui,
        sample_records_ui,
        log_file_ui,
        max_clusters_to_test_ui,
        max_lat_ui,
        max_lon_ui,
        max_taxa_enabled_ui,
        max_taxa_value_ui,
        min_clusters_to_test_ui,
        min_geocode_presence_enabled_ui,
        min_geocode_presence_value_ui,
        min_hex_records_ui,
        min_lat_ui,
        min_lon_ui,
        no_images,
        no_seed,
        no_stop,
        parquet_source_path_ui,
        run_button_ui,
        seed_ui,
        taxon_scope_ui,
        terrestrial_only,
        no_hex_floor,
        no_findings,
        findings_output,
        composition_metric,
        default_display_level,
        hierarchy_levels,
        linkage,
        reduction,
        metric_weights,
        num_clusters_pinned,
    )


@app.cell(hide_code=True)
def _(
    folium,
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
    min_hex_records_ui,
    min_lat_ui,
    min_lon_ui,
    mo,
    parquet_source_path_ui,
    sample_records_ui,
    seed_ui,
    taxon_scope_ui,
):
    def build_map():
        m = folium.Map(
            tiles="Esri.WorldGrayCanvas",
        )
        bounds = [
            [min_lat_ui.value, min_lon_ui.value],
            [max_lat_ui.value, max_lon_ui.value],
        ]
        folium.Rectangle(bounds=bounds).add_to(m)
        m.fit_bounds(bounds, padding=[20, 20])
        return m

    data_source_box = mo.vstack([
        mo.md("### 📁 Data Source & Spatial Resolution"),
        mo.hstack([parquet_source_path_ui, log_file_ui], widths="equal"),
        mo.hstack([geocode_precision_ui, min_hex_records_ui], widths="equal"),
        mo.md("#### Geographic Extent & Interactive Preview"),
        mo.hstack(
            [
                mo.vstack([
                    mo.md("**Minimum Coordinates**"),
                    min_lat_ui,
                    min_lon_ui,
                    mo.md("**Maximum Coordinates**"),
                    max_lat_ui,
                    max_lon_ui,
                ]),
                build_map(),
            ],
            widths="equal",
        ),
    ])

    sampling_taxa_box = mo.vstack([
        mo.md("### 🔬 Taxonomic Scope & Record Sampling"),
        taxon_scope_ui,
        mo.md("**Record Capping & Uniform Sampling:**"),
        mo.hstack([limit_results_enabled_ui, limit_results_value_ui, sample_records_ui]),
        mo.md("---"),
        mo.md(
            "**Taxa Filtering:** Filter uninformative or rare taxa to improve performance "
            "and reduce noise before calculating ecological dissimilarity."
        ),
        mo.hstack([max_taxa_enabled_ui, max_taxa_value_ui]),
        mo.md("_Keep top N taxa by occurrence count (recommended: 5,000–10,000)._"),
        mo.hstack([min_geocode_presence_enabled_ui, min_geocode_presence_value_ui]),
        mo.md("_Minimum fraction of hexagons a taxon must appear in (e.g., 0.05 = 5%)._"),
    ])

    clustering_box = mo.vstack([
        mo.md("### 🧩 Clustering Optimization Range"),
        mo.md(
            "Configure the range of cluster counts ($k$) to evaluate. Metrics "
            "(Silhouette, Calinski-Harabasz, Davies-Bouldin) will be calculated for each $k$."
        ),
        mo.hstack([min_clusters_to_test_ui, max_clusters_to_test_ui, seed_ui]),
    ])

    return mo.accordion({
        "📁 Data Source & Spatial Extent": data_source_box,
        "🔬 Taxonomic Scope & Sampling": sampling_taxa_box,
        "🧩 Clustering Parameters": clustering_box,
    })


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
    sample_records_ui,
    log_file_ui,
    max_clusters_to_test_ui,
    max_lat_ui,
    max_lon_ui,
    max_taxa_enabled_ui,
    max_taxa_value_ui,
    min_clusters_to_test_ui,
    min_geocode_presence_enabled_ui,
    min_geocode_presence_value_ui,
    min_hex_records_ui,
    min_lat_ui,
    min_lon_ui,
    mo,
    no_seed,
    no_stop,
    parquet_source_path_ui,
    run_button_ui,
    seed_ui,
    taxon_scope_ui,
    terrestrial_only,
    no_hex_floor,
    composition_metric,
    hierarchy_levels,
    linkage,
    reduction,
    metric_weights,
    num_clusters_pinned,
):
    from src.taxon_scope import parse_scope
    from src.types import Bbox

    # Resolve final values from UI elements
    sample_records = int(sample_records_ui.value) or None
    limit_results = (
        limit_results_value_ui.value if limit_results_enabled_ui.value else None
    )
    # A uniform sample supersedes the scan-order cap rather than stacking with
    # it; --limit-results is on by default, so without this a --sample-records
    # run would draw its sample from the first 1000 rows.
    if sample_records is not None:
        limit_results = None
    log_file = log_file_ui.value
    parquet_source_path = parquet_source_path_ui.value
    min_lat = min_lat_ui.value
    max_lat = max_lat_ui.value
    min_lon = min_lon_ui.value
    max_lon = max_lon_ui.value
    # Raises ValueError on an unparseable or unknown scope, which surfaces as a
    # cell error rather than silently running unscoped.
    taxon_scope = parse_scope(taxon_scope_ui.value)
    geocode_precision = geocode_precision_ui.value
    min_clusters_to_test = min_clusters_to_test_ui.value
    max_clusters_to_test = max_clusters_to_test_ui.value
    max_taxa = max_taxa_value_ui.value if max_taxa_enabled_ui.value else None
    min_geocode_presence = (
        min_geocode_presence_value_ui.value
        if min_geocode_presence_enabled_ui.value
        else None
    )
    bounding_box = Bbox.from_coordinates(min_lat, max_lat, min_lon, max_lon)
    random_seed = None if no_seed else int(seed_ui.value)
    min_hex_records = int(min_hex_records_ui.value) or None

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
            {
                "variable": "taxon_scope",
                "value": str(taxon_scope) if taxon_scope else "(all taxa)",
            },
            {"variable": "geocode_precision", "value": geocode_precision},
            {"variable": "min_clusters_to_test", "value": min_clusters_to_test},
            {"variable": "max_clusters_to_test", "value": max_clusters_to_test},
            {"variable": "max_taxa", "value": max_taxa},
            {"variable": "min_geocode_presence", "value": min_geocode_presence},
            {"variable": "random_seed", "value": random_seed},
            {"variable": "min_hex_records", "value": min_hex_records},
            {"variable": "terrestrial_only", "value": terrestrial_only},
            {"variable": "no_hex_floor", "value": no_hex_floor},
            {"variable": "composition_metric", "value": composition_metric},
            {"variable": "hierarchy_levels", "value": hierarchy_levels},
            {"variable": "linkage", "value": linkage},
            {"variable": "reduction", "value": reduction},
            {"variable": "metric_weights", "value": str(metric_weights)},
            {"variable": "num_clusters_pinned", "value": num_clusters_pinned},
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
        min_hex_records,
        parquet_source_path,
        random_seed,
        sample_records,
        taxon_scope,
        terrestrial_only,
        no_hex_floor,
        composition_metric,
        linkage,
        reduction,
        metric_weights,
        num_clusters_pinned,
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

    logging.basicConfig(
        filename=log_file,
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        encoding="utf-8",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Step 1. Fetch & Preprocess Data

    In this step, species occurrence records are loaded and filtered:
    1. **Darwin Core Ingestion**: Reads occurrence records filtered by bounding box and optional taxonomic scope.
    2. **Terrestrial Masking**: (Optional) Drops marine hexagons using Natural Earth coastline polygons.
    3. **Parquet Materialization**: Intermediate LazyFrames are materialized to local disk cache to prevent redundant scans over raw GBIF files.
    4. **Sampling Floor Filtering**: Discard under-sampled hexagons (below a data-derived threshold) to prevent Ward linkage from peeling sparse cells off as singleton clusters.
    5. **Taxa Frequency Filtering**: Filters rare or uninformative taxa to accelerate downstream matrix operations.
    """)
    return


@app.cell
def _(
    bounding_box,
    defaults,
    geocode_precision,
    limit_results,
    materialize_parquet,
    min_hex_records,
    parquet_source_path,
    random_seed,
    sample_records,
    taxon_scope,
    terrestrial_only,
    no_hex_floor,
):
    from src.dataframes.darwin_core import build_darwin_core_lf
    from src.geocode import (
        adaptive_min_hex_records,
        filter_sparse_geocodes_lf,
        filter_terrestrial_geocodes_lf,
    )
    from src.logging import logger

    darwin_core_lf = build_darwin_core_lf(
        source_path=parquet_source_path,
        bounding_box=bounding_box,
        limit=limit_results,
        scope=taxon_scope,
        sample_records=sample_records,
        # --no-seed leaves random_seed None; that opts out of seeding UMAP for
        # the sake of threading, not out of knowing which records a run read.
        seed=random_seed if random_seed is not None else 0,
    )

    # Applied here, upstream of the geocode set, so that the geocodes and the
    # taxa counts are both derived from the same rows.
    if terrestrial_only:
        darwin_core_lf = filter_terrestrial_geocodes_lf(
            darwin_core_lf, geocode_precision
        )

    # Spill once, here, rather than letting three downstream stages each re-read
    # the source. Snapshot scans cannot be pruned, so every consumer of this
    # frame -- build_geocode_lf, build_taxonomy_lf, build_geocode_taxa_counts_lf
    # -- otherwise pays for a full pass. On the published East Coast run that
    # was roughly 17 of the notebook's 28 minutes, against about one minute for
    # all the clustering downstream of it.
    darwin_core_lf = materialize_parquet(darwin_core_lf, cache_key="DarwinCoreSchema")

    # The sampling floor is derived and applied *after* the spill, deliberately.
    # Both steps read every row -- one to find the median hexagon, one to drop
    # the hexagons below it -- and doing that upstream put two more full passes
    # over the source in front of the spill that exists to prevent exactly that.
    # On the published run it took the job from about 30 minutes to 55 and then
    # the runner was killed for memory. Downstream of the spill both passes read
    # a local parquet.
    #
    # Without a floor, Ward peels under-sampled hexagons off as singleton
    # clusters and the partition stops surviving perturbation: hiding 5% of
    # records took three of four test regions to chance agreement.
    if not no_hex_floor:
        floor = min_hex_records
        if floor is None:
            floor = adaptive_min_hex_records(
                darwin_core_lf,
                geocode_precision,
                defaults.MIN_HEX_RECORDS_ABSOLUTE_FLOOR,
                defaults.MIN_HEX_RECORDS_MEDIAN_FRACTION,
                defaults.MIN_HEX_RECORDS_CEILING,
            )
            logger.info(
                f"Sampling floor derived from the data: {floor} records per hexagon"
            )
        darwin_core_lf = materialize_parquet(
            filter_sparse_geocodes_lf(darwin_core_lf, geocode_precision, floor),
            cache_key="DarwinCoreFilteredSchema",
        )
    return (darwin_core_lf,)


@app.cell
def _(bounding_box, materialize_parquet, darwin_core_lf, geocode_precision):
    from src.dataframes.geocode import build_geocode_lf

    geocode_lf_with_edges = materialize_parquet(
        build_geocode_lf(
            darwin_core_lf,
            geocode_precision,
            bounding_box=bounding_box,
        ),
        cache_key="GeocodeSchema",
    )
    return (geocode_lf_with_edges,)


@app.cell
def _(materialize_parquet, geocode_lf_with_edges):
    from src.dataframes.geocode import build_geocode_no_edges_lf

    geocode_unfiltered_lf = materialize_parquet(
        build_geocode_no_edges_lf(
            geocode_lf_with_edges,
        ),
        cache_key="GeocodeNoEdgesSchema",
    )
    return (geocode_unfiltered_lf,)


@app.cell
def _(geocode_lf_with_edges):
    from src.dataframes.geocode_neighbors import build_geocode_neighbors_df

    # Build neighbors for all geocodes (including edges)
    geocode_neighbors_with_edges_df = build_geocode_neighbors_df(
        geocode_lf_with_edges.collect(),
    )
    return (geocode_neighbors_with_edges_df,)


@app.cell
def _(materialize_parquet, geocode_lf, geocode_neighbors_with_edges_df):
    from src.dataframes.geocode_neighbors import build_geocode_neighbors_no_edges_df

    # Build neighbors for filtered geocodes only
    geocode_neighbors_df = materialize_parquet(
        build_geocode_neighbors_no_edges_df(
            geocode_neighbors_with_edges_df,
            geocode_lf.collect(),
        ),
        cache_key="GeocodeNeighborsSchema",
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
    materialize_parquet,
    darwin_core_lf,
    geocode_precision,
    geocode_unfiltered_lf,
):
    from src.dataframes.taxonomy import build_taxonomy_lf

    taxonomy_lf = materialize_parquet(
        build_taxonomy_lf(
            darwin_core_lf,
            geocode_precision,
            geocode_unfiltered_lf,
            bounding_box=bounding_box,
        ),
        cache_key="TaxonomySchema",
    )
    return (taxonomy_lf,)


@app.cell
def _(materialize_parquet, darwin_core_lf, taxonomy_lf):
    from src.dataframes.taxon_clade import build_taxon_clade_lf

    # One row per taxon, so it is small enough to spill and hold. `None` when
    # the source carried no rank columns; the findings page then says the clade
    # questions were not answered rather than guessing at them.
    _clade_lf = build_taxon_clade_lf(darwin_core_lf, taxonomy_lf)
    taxon_clade_lf = (
        None
        if _clade_lf is None
        else materialize_parquet(_clade_lf, cache_key="TaxonCladeSchema")
    )
    return (taxon_clade_lf,)


@app.cell
def _(taxonomy_lf):
    taxonomy_lf.limit(100).collect(engine="streaming")
    return


@app.cell
def _(
    bounding_box,
    materialize_parquet,
    darwin_core_lf,
    geocode_precision,
    geocode_unfiltered_lf,
    taxonomy_lf,
):
    from src.dataframes.geocode_taxa_counts import build_geocode_taxa_counts_lf

    geocode_taxa_counts_unfiltered_lf = materialize_parquet(
        build_geocode_taxa_counts_lf(
            darwin_core_lf,
            geocode_precision,
            taxonomy_lf,
            geocode_unfiltered_lf,
            bounding_box=bounding_box,
        ),
        cache_key="GeocodeTaxaCountsSchema",
    )
    return (geocode_taxa_counts_unfiltered_lf,)


@app.cell
def _(geocode_taxa_counts_unfiltered_lf, max_taxa, min_geocode_presence):
    from src.dataframes.geocode_taxa_counts import filter_top_taxa_lf

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
def _(geocode_taxa_counts_lf, geocode_unfiltered_lf, pl):
    # Filter geocode_unfiltered_lf to only include geocodes present in geocode_taxa_counts_lf
    geocode_lf = geocode_unfiltered_lf.join(
            geocode_taxa_counts_lf.select(pl.col("geocode").unique()),
            on="geocode",
            how="semi",
        )
    return (geocode_lf,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Step 2. Build Matrices & Perform Clustering

    In this step, spatial graph connectivity and ecological dissimilarity matrices are constructed for all valid H3 hexagons:
    1. **Spatial Connectivity Matrix (`GeocodeConnectivityMatrix`)**: Constructs spatial adjacency graphs across H3 hexagons to enforce spatial contiguity during agglomeration.
    2. **Ecological Distance Matrix (`GeocodeDistanceMatrix`)**: Calculates species turnover ($\beta_{\text{sim}}$, Sørensen, or Bray-Curtis) reduced via PCoA or UMAP for Euclidean compatibility.
    3. **Agglomerative Spatial Clustering**: Performs Ward linkage clustering constrained by spatial contiguity across a range of cluster counts $k \in [k_{\min}, k_{\max}]$.
    4. **Multi-Metric Cluster Evaluation**: Scores candidate partitions using Silhouette, Calinski-Harabasz, and Davies-Bouldin indices.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Spatial Graph Connectivity (`GeocodeConnectivityMatrix`)
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
    ## Ecological Composition Distance (`GeocodeDistanceMatrix`)
    """)
    return


@app.cell
def _(geocode_lf, geocode_taxa_counts_lf, mo, np, random_seed, composition_metric, reduction):
    from src.matrices.geocode_distance import GeocodeDistanceMatrix

    geocode_distance_matrix = GeocodeDistanceMatrix.build(
        geocode_taxa_counts_lf,
        geocode_lf,
        random_state=random_seed,
        metric=composition_metric,
        reduction=reduction,
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
    ## Cluster Hierarchy & Optimization
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1. Compute Agglomerative Hierarchy Across $k \in [k_{\min}, k_{\max}]$
    """)
    return


@app.cell
def _(
    materialize_parquet,
    geocode_connectivity_matrix,
    geocode_distance_matrix,
    geocode_lf,
    linkage,
    max_clusters_to_test,
    min_clusters_to_test,
):
    from src.dataframes.geocode_cluster import build_geocode_cluster_multi_k_df

    all_clusters_df = materialize_parquet(
        build_geocode_cluster_multi_k_df(
            geocode_lf,
            geocode_distance_matrix,
            geocode_connectivity_matrix,
            min_k=min_clusters_to_test,
            max_k=max_clusters_to_test,
            linkage=linkage,
        ),
        cache_key="GeocodeClusterMultiKSchema",
    ).collect(engine="streaming")
    return (all_clusters_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Score every cut

    The score below is a diagnostic, not the decision. Which cut is published
    is settled in the next cell.
    """)
    return


@app.cell
def _(
    all_clusters_df,
    geocode_distance_matrix,
    metric_weights,
    num_clusters_pinned,
):
    from src.cluster_optimization import optimize_num_clusters

    optimal_num_clusters, all_cluster_metrics = optimize_num_clusters(
        geocode_distance_matrix,
        all_clusters_df,
        weights=metric_weights,
        pinned_k=num_clusters_pinned,
    )

    all_cluster_metrics
    return all_cluster_metrics, optimal_num_clusters


@app.cell
def _(all_cluster_metrics, materialize_parquet):
    # Cache the results

    all_cluster_metrics_df = materialize_parquet(
        all_cluster_metrics,
        cache_key="GeocodeClusterMetricsSchema",
    ).collect(engine="streaming")
    return (all_cluster_metrics_df,)


@app.cell
def _(
    default_display_level,
    hierarchy_levels,
    max_clusters_to_test,
    min_clusters_to_test,
    optimal_num_clusters,
):
    from src.hierarchy import default_ladder, resolve_default_level, resolve_levels

    # Resolved here rather than at the writer, because everything below is built
    # at `published_level`. The selector's k is kept and reported, but it stops
    # deciding what the outputs describe: it maximises a score silhouette
    # dominates, silhouette falls monotonically with k on these distances, and
    # the cut it lands on scored last against both references the run computes.
    # See defaults.DEFAULT_DISPLAY_LEVEL.
    hierarchy_level_list = resolve_levels(
        hierarchy_levels
        or default_ladder(min_clusters_to_test, max_clusters_to_test),
        optimal_num_clusters,
        min_clusters_to_test,
        max_clusters_to_test,
        display=default_display_level,
    )
    published_level = resolve_default_level(
        default_display_level, optimal_num_clusters, hierarchy_level_list
    )
    return hierarchy_level_list, published_level


@app.cell(hide_code=True)
def _(hierarchy_level_list, mo, optimal_num_clusters, published_level):
    # Stated before the metric plots below, which mark the selector's peak.
    # Without this the notebook shows the selector's k in four places and the
    # published level in none, so a reader reasonably concludes the run
    # published the selector's cut. It does not.
    mo.md(
        f"""
    ### Publishing {published_level} regions

    Levels emitted: **{", ".join(str(k) for k in hierarchy_level_list)}**.
    Everything built for a single cut below — the GeoJSON, the per-cluster taxa
    statistics, the cluster colours, the PERMANOVA — describes
    **{published_level}**.

    The selector's combined score peaked at **{optimal_num_clusters}**, which is
    reported as a diagnostic and does not decide anything. It maximises a score
    silhouette dominates, and silhouette falls monotonically with k on saturated
    ecological distances, so its peak is the bottom of the tested range whatever
    the data says. On the published run that cut scored last against both
    references this notebook computes: the two clades agreed no better than
    chance there, and agreement with EPA Level II was its lowest. See the
    Findings section.
    """
        if published_level != optimal_num_clusters
        else f"""
    ### Publishing {published_level} regions

    Levels emitted: **{", ".join(str(k) for k in hierarchy_level_list)}**.
    The selector's combined score also peaked here.
    """
    )
    return


@app.cell
def _(all_clusters_df, materialize_parquet, published_level):
    # Create base GeocodeClusterSchema (single k) for downstream use
    from src.dataframes.geocode_cluster import build_geocode_cluster_df

    geocode_cluster_df = materialize_parquet(
        build_geocode_cluster_df(
            all_clusters_df,
            published_level,
        ),
        cache_key="GeocodeClusterSchema",
    ).collect(engine="streaming")
    return (geocode_cluster_df,)


@app.cell
def _(all_cluster_metrics_df):
    all_cluster_metrics_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Optimization Results Table
    """)
    return


@app.cell
def _(all_cluster_metrics_df):
    from src.dataframes.geocode_cluster_metrics import get_metrics_summary

    get_metrics_summary(all_cluster_metrics_df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Multi-Metric Cluster Validation
    """)
    return


@app.cell
def _(all_cluster_metrics_df, optimal_num_clusters):
    from src.plot.cluster_metrics import plot_all_metrics_vs_k

    plot_all_metrics_vs_k(
        all_cluster_metrics_df,
        optimal_k=optimal_num_clusters,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Normalized Metrics Comparison
    """)
    return


@app.cell
def _(all_cluster_metrics_df, optimal_num_clusters):
    from src.plot.cluster_metrics import plot_normalized_metrics

    plot_normalized_metrics(
        all_cluster_metrics_df,
        optimal_k=optimal_num_clusters,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## `ClusterNeighbors`
    """)
    return


@app.cell
def _(materialize_parquet, geocode_cluster_df, geocode_neighbors_df):
    from src.dataframes.cluster_neighbors import build_cluster_neighbors_df

    cluster_neighbors_lf = materialize_parquet(
        build_cluster_neighbors_df(
            geocode_neighbors_df,
            geocode_cluster_df,
        ),
        cache_key="ClusterNeighborsSchema",
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
def _(materialize_parquet, geocode_cluster_df, geocode_taxa_counts_lf, taxonomy_lf):
    from src.dataframes.cluster_taxa_statistics import build_cluster_taxa_statistics_df

    cluster_taxa_statistics_df = materialize_parquet(
        build_cluster_taxa_statistics_df(
            geocode_taxa_counts_lf,
            geocode_cluster_df.lazy(),
            taxonomy_lf,
        ),
        cache_key="ClusterTaxaStatisticsSchema",
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
def _(materialize_parquet, cluster_neighbors_lf, cluster_taxa_statistics_df):
    from src.dataframes.cluster_significant_differences import build_cluster_significant_differences_df

    cluster_significant_differences_df = materialize_parquet(
        build_cluster_significant_differences_df(
            cluster_taxa_statistics_df,
            cluster_neighbors_lf,
        ),
        cache_key="ClusterSignificantDifferencesSchema",
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
def _(materialize_parquet, geocode_cluster_df, geocode_lf):
    from src.dataframes.cluster_boundary import build_cluster_boundary_df

    cluster_boundary_df = materialize_parquet(
        build_cluster_boundary_df(
            geocode_cluster_df,
            geocode_lf,
        ),
        cache_key="ClusterBoundarySchema",
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
def _(published_level):
    # Use taxonomic coloring if we have at least 10 clusters, otherwise use geographic
    color_method = "taxonomic" if published_level >= 10 else "geographic"
    return (color_method,)


@app.cell
def _(
    materialize_parquet,
    cluster_neighbors_lf,
    cluster_taxa_statistics_df,
    color_method,
):
    from src.dataframes.cluster_color import build_cluster_color_df

    cluster_colors_df = materialize_parquet(
        build_cluster_color_df(
            cluster_neighbors_lf,
            cluster_taxa_statistics_df,
            color_method=color_method,
        ),
        cache_key="ClusterColorSchema",
    ).collect(engine="streaming")
    return (cluster_colors_df,)


@app.cell
def _(cluster_colors_df):
    cluster_colors_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Step 3. Region Diagnostics & Validation Analyses

    In this step, the quality, distinctiveness, and spatial coherence of the derived bioregions are evaluated:
    1. **PERMANOVA**: Permutational Multivariate Analysis of Variance tests whether species composition differs significantly across the identified bioregions.
    2. **Hexagon Silhouette Analysis**: Measures how strongly each individual hexagon belongs to its assigned bioregion relative to neighboring clusters.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## PERMANOVA Compositional Significance Test (`PermanovaResults`)
    """)
    return


@app.cell
def _(
    materialize_parquet,
    geocode_cluster_df,
    geocode_distance_matrix,
    geocode_lf,
    random_seed,
):
    from src.dataframes.permanova_results import build_permanova_results_df

    permanova_results_df = materialize_parquet(
        build_permanova_results_df(
            geocode_distance_matrix=geocode_distance_matrix,
            geocode_cluster_df=geocode_cluster_df,
            geocode_lf=geocode_lf,
            seed=random_seed,
        ),
        cache_key="PermanovaResultsSchema",
    ).collect(engine="streaming")
    return (permanova_results_df,)


@app.cell
def _(permanova_results_df):
    permanova_results_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Hexagon Silhouette Score Analysis (`GeocodeSilhouetteScore`)
    """)
    return


@app.cell
def _(all_clusters_df, geocode_distance_matrix, published_level, pl):
    from src.dataframes.geocode_silhouette_score import build_geocode_silhouette_score_df

    # Get clustering for optimal k
    k_df = all_clusters_df.filter(pl.col("num_clusters") == published_level)

    geocode_silhouette_score_df = build_geocode_silhouette_score_df(
        geocode_distance_matrix, k_df
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
    # Step 4. Export Outputs & Interactive Visualizations

    In this final step, the spatial boundaries, ordination plots, representative species images, and analytical findings are exported:
    1. **GeoJSON Boundaries**: Polygons with cluster properties and colors for interactive web maps.
    2. **Ordination Plot**: Ordination scatter plot showing cluster separation in reduced composition space.
    3. **Representative Species Images**: Significant indicator taxa with species images retrieved from Wikidata.
    4. **Analytical Findings Page**: Detailed evaluation report including sampling effort bias, clade congruence, and ecoregion framework comparisons.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. GeoJSON Feature Collection Generation
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
    ## Clustermap Visualization (Indicator Taxa)
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
        result = mo.md("_No significant indicator taxa differences found between clusters._")
    else:
        result = heatmap.figure

    result


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## `SignificantTaxaImages`
    """)
    return


@app.cell
def _(
    materialize_parquet,
    cluster_significant_differences_df,
    no_images,
    taxonomy_lf,
):
    from src.dataframes.significant_taxa_images import build_significant_taxa_images_df

    significant_taxa_images_df = materialize_parquet(
        build_significant_taxa_images_df(
            cluster_significant_differences_df,
            taxonomy_lf.collect(engine="streaming"),
            fetch_images=not no_images,
        ),
        cache_key="SignificantTaxaImagesSchema",
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Findings

    What this run found, as opposed to what it computed. Every figure below is
    derived from this run's own outputs, so it stays honest when the parameters
    change.

    Two results this project has established are **not** here, because neither can
    be computed from a single run: clade congruence needs one run per clade, and
    the Sorensen/betasim comparison needs one run per index.
    """)
    return


@app.cell(hide_code=True)
def _(geocode_taxa_counts_lf, mo):
    from src.plot.findings import effort_vs_richness

    _chart, _rho = effort_vs_richness(geocode_taxa_counts_lf)
    # Rendered bare rather than through mo.ui.altair_chart. The wrapper exists
    # to send selections back to Python, which a static export has no kernel to
    # receive, and on the bar chart below it drops the marks entirely -- axes
    # and titles draw, the bars do not.
    mo.vstack([
        _chart,
        mo.md(
            f"Records and taxa correlate at **{_rho:.3f}**. The closer this is to 1, "
            "the more a hexagon's apparent richness is a record of visits rather than "
            "of biota — and the more the composition metric has to do to see past it."
        ),
    ])
    return


@app.cell(hide_code=True)
def _(geocode_distance_matrix, geocode_taxa_counts_lf, mo):
    from src.plot.findings import dissimilarity_vs_effort

    # The composition distances, not the ones clustering ran on: this asks what
    # the *index* does with uneven sampling, before any reduction.
    _composition = geocode_distance_matrix.abundance_condensed()
    if _composition is None:
        _out = mo.md("_No composition distances retained for this metric._")
    else:
        _chart, _rho = dissimilarity_vs_effort(_composition, geocode_taxa_counts_lf)
        if _chart is None:
            _out = mo.md(
                "_Too few hexagons to bin by sampling effort — this run cannot "
                "answer whether the index is tracking it._"
            )
        else:
            _out = mo.vstack([
                _chart,
                mo.md(
                    f"Dissimilarity tracks the effort gap at **{_rho:.3f}**. Rising "
                    "bars mean hexagons are being called different partly because "
                    "one was visited more often than the other."
                ),
            ])
    _out
    return


@app.cell(hide_code=True)
def _(all_cluster_metrics_df, mo, optimal_num_clusters, published_level):
    from src.plot.findings import metrics_by_k

    metrics_by_k(
        all_cluster_metrics_df,
        published_k=published_level,
        selector_k=optimal_num_clusters,
    )
    return


@app.cell
def _(
    all_clusters_df,
    default_display_level,
    geocode_lf,
    geocode_neighbors_df,
    geocode_taxa_counts_lf,
    hierarchy_level_list,
    no_images,
    published_level,
    taxonomy_lf,
):
    from src.hierarchy import build_hierarchy_json
    from src.output import prepare_file_path

    # Every level reruns the chain the cells above ran for the selected one, so
    # the selected level's entry is what the single-level writer produced; the
    # others are the remaining cuts of the same tree. The cells above are kept
    # because they are what the notebook displays.
    _json = build_hierarchy_json(
        all_clusters_df,
        geocode_lf,
        geocode_neighbors_df,
        geocode_taxa_counts_lf,
        taxonomy_lf,
        levels=hierarchy_level_list,
        default_level=published_level,
        fetch_images=not no_images,
    )
    with open(prepare_file_path("frontend/aggregations.json"), "w") as _writer:
        _writer.write(_json)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Findings

    The same numbers as the charts above, plus the two this run has to work for
    -- clade congruence and agreement with a published framework -- written to
    a standalone page beside the other outputs.
    """)
    return


@app.cell
def _(
    all_clusters_df,
    bounding_box,
    composition_metric,
    default_display_level,
    geocode_lf,
    geocode_precision,
    geocode_taxa_counts_lf,
    max_clusters_to_test,
    min_clusters_to_test,
    findings_output,
    mo,
    no_findings,
    optimal_num_clusters,
    parquet_source_path,
    published_level,
    random_seed,
    reduction,
    taxon_clade_lf,
    taxonomy_lf,
):
    import polars as _pl

    # Aliased: marimo requires each global to be defined in exactly one cell,
    # and the hierarchy cell above already imports this name.
    from src.findings import build_findings_data as _build_findings_data
    from src.findings_page import RunContext as _RunContext
    from src.findings_page import findings_summary_json as _findings_summary_json
    from src.findings_page import write_findings_page as _write_findings_page
    from src.output import prepare_file_path as _prepare_file_path

    if no_findings:
        _out = mo.md("_Findings page skipped (`--no-findings`)._")
    else:
        _context = _RunContext(
            source=str(parquet_source_path),
            bbox=(
                f"{bounding_box.sw.lat:g}-{bounding_box.ne.lat:g}N, "
                f"{bounding_box.sw.lng:g}-{bounding_box.ne.lng:g}E"
            ),
            geocode_precision=geocode_precision,
            hexagons=geocode_lf.select(_pl.len()).collect().item(),
            taxa=taxonomy_lf.select(_pl.len()).collect().item(),
            # The clade shares are fractions of this, not of the taxonomy: the
            # taxa filters run before clustering, so most of the taxonomy is
            # not in the map at all.
            taxa_analysed=geocode_taxa_counts_lf.select(
                _pl.col("taxonId").n_unique()
            )
            .collect(engine="streaming")
            .item(),
            records=geocode_taxa_counts_lf.select(_pl.col("count").sum())
            .collect(engine="streaming")
            .item(),
            chosen_k=optimal_num_clusters,
            composition_metric=composition_metric,
            seed=random_seed,
            display_k=published_level,
        )
        _data = _build_findings_data(
            _context,
            geocode_taxa_counts_lf,
            geocode_lf,
            all_clusters_df,
            taxon_clade_lf,
            min_k=min_clusters_to_test,
            max_k=max_clusters_to_test,
            seed=random_seed,
            metric=composition_metric,
            reduction=reduction,
        )
        _write_findings_page(_data, _prepare_file_path(findings_output))
        # The same numbers as JSON, so a reader who wants to check one does not
        # have to scrape the page for it.
        _json_path = findings_output.removesuffix(".html") + ".json"
        with open(_prepare_file_path(_json_path), "w") as _writer:
            _writer.write(_findings_summary_json(_data))
        _out = mo.md(
            f"Wrote `{findings_output}`. "
            f"{len(_data.congruence)} congruence points, "
            f"{len(_data.reference_by_k)} reference cuts, "
            f"{len(_data.clade_shares)} clade shares"
            + (f", {len(_data.skipped)} section(s) skipped." if _data.skipped else ".")
        )
    _out
    return


if __name__ == "__main__":
    app.run()
