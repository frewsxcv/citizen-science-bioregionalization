# bioregion_rs

Rust core for the citizen-science bioregionalization pipeline. See
`../RUST_MIGRATION_PLAN.md` for the overall strategy. This crate is the Phase 0
scaffold: it establishes the Rust↔Python interop and the correctness harness that
every subsequent file migration will use.

## What's here

- `src/lib.rs` — a PyO3 extension module (`bioregion_rs`) exposing:
  - `darken_hex_color(hex, factor=0.5)` — pure function (plain PyO3 boundary),
    mirrors `src/colors.py`.
  - `select_geocode`/`with_geocode`/`filter_by_bounding_box` — mirrors `src/geocode.py`.
  - `build_geocode` — mirrors `src/dataframes/geocode.py`.
  - `build_taxonomy` — mirrors `src/dataframes/taxonomy.py`.
  - `build_geocode_taxa_counts` — mirrors `src/dataframes/geocode_taxa_counts.py`.
  - `build_geocode_neighbors`/`build_geocode_neighbors_no_edges` — mirrors
    `src/dataframes/geocode_neighbors.py`.
  - `build_geocode_connectivity_matrix` — mirrors `src/matrices/geocode_connectivity.py`.
  - `build_cluster_taxa_statistics` — mirrors `src/dataframes/cluster_taxa_statistics.py`.
  - `build_cluster_neighbors` — mirrors `src/dataframes/cluster_neighbors.py`.
  - `build_cluster_distance_matrix` — mirrors `src/matrices/cluster_distance.py`.
  - `build_cluster_color` — mirrors `src/dataframes/cluster_color.py` (geographic
    path only; the taxonomic path uses UMAP + MDS and stays in Python).
  - `build_cluster_boundary` — mirrors `src/dataframes/cluster_boundary.py`.
  - `build_cluster_significant_differences` — mirrors
    `src/dataframes/cluster_significant_differences.py`, including a from-scratch
    Fisher's exact test port (no external stats crate).
  - `build_permanova_results` — mirrors `src/dataframes/permanova_results.py`. The
    pseudo-F statistic is exact; the p-value uses Rust's own RNG for its Monte Carlo
    permutation test (the Python call site is itself unseeded, so bit-reproducibility
    isn't achievable or expected — see the plan for details).
  - `build_geocode_cluster_metrics`/`select_optimal_k_elbow` — mirrors
    `src/dataframes/geocode_cluster_metrics.py`. Elbow detection uses the `kneed`
    crate rather than a hand-rolled Kneedle port — see the plan for why (a hand-rolled
    first pass had a real, if subtle, bug around curve-endpoint handling).
  - `build_geocode_silhouette_score` — mirrors `src/dataframes/geocode_silhouette_score.py`
    (per-point silhouette scores; same formula as `build_geocode_cluster_metrics`'s
    mean, extended to per-geocode).
  - `optimize_num_clusters` — mirrors `src/cluster_optimization.py`; thin in-process
    orchestration over `build_geocode_cluster_metrics` + `find_elbow_point`.
  - `build_geojson_feature_collection` — mirrors `src/geojson.py`. Returns the
    FeatureCollection as a JSON string (no `serde_json` dependency; the shape here is
    small and hand-built via `format!`), not a `geojson.FeatureCollection` object —
    see the plan for why that's a safe stand-in.
- `harness.py` — runs each Rust function and its Python counterpart on the same
  input and asserts they match. The template for migrating each file.

CI (`.github/workflows/bioregion-rs.yml`) runs `cargo test --lib`, builds the
extension via `maturin develop`, and runs `harness.py` on every push.

## Build & test

```bash
# Rust unit tests (pure logic; needs a Python on PATH for pyo3 linking):
cargo test --lib

# Build the extension into the project venv:
uv run maturin develop --manifest-path bioregion_rs/Cargo.toml

# Correctness / interop harness (Rust output vs Python output):
uv run python bioregion_rs/harness.py
```

## Version pinning (important)

Python-side `polars` and Rust-side `polars` must stay ABI-compatible; bump them
together.

| Component | Version | Notes |
|---|---|---|
| Python `polars` | 1.37.0 | the target ABI |
| Rust `polars` | 0.54.4 | matches Python 1.37 |
| `pyo3-polars` | 0.27 | pins Rust polars to 0.54 |
| `pyo3` | 0.28 | abi3-py313 |
| `h3o` | 0.8 | pure-Rust H3; produces cell ids identical to `polars-h3` |
| `geo` | 0.33.1 | boolean ops (`unary_union`) for `cluster_boundary` |
| `rand` | 0.9 | PERMANOVA's Monte Carlo shuffle; already resolved transitively by `polars` |
| `kneed` | 1.0 | Kneedle elbow detection; independent Rust port of the Python `kneed` package — pulled in for correctness (see the plan), at the cost of a real added dependency tree (`nalgebra`, `glam`, `polyfit-rs`) for a polynomial-interpolation path this crate doesn't use |

### Known toolchain constraint

polars 0.54.4's `lazy` feature set (`polars-lazy` / `polars-stream`) does **not**
compile on the current nightly toolchain due to internal feature-unification bugs.
Phase 0 therefore uses only the **eager DataFrame API**. This is sufficient for the
robust interop path (Arrow-FFI DataFrame transfer + the parquet stage boundary).
LazyFrame plan-passing (`PyLazyFrame`) is deferred — it is the version-sensitive
path the migration plan already flags, and stages hand off via parquet in the
meantime. Re-enable `lazy` (add `"lazy"` to the `polars` features and `pyo3-polars`,
restore a `PyLazyFrame` function) once a polars/toolchain combo builds it cleanly.

## `cargo test` + PyO3 note

`pyo3/extension-module` is set only in `[tool.maturin]`, not in the crate's default
features, because it suppresses libpython linking and breaks `cargo test`'s
standalone binary. If `cargo test` can't find a Python to link against at *build*
time, pass `PYO3_PYTHON=$(uv run which python)`.

Separately, at *run* time the test binary dynamically links libpython but doesn't
get an rpath to uv's managed Python install, so the runtime loader may not find it
either (`error while loading shared libraries: libpython3.13.so.1.0` on Linux,
`Library not loaded: .../libpython3.13.dylib` on macOS). Point the loader at it
explicitly:

```bash
PYTHON_LIBDIR=$(uv run python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
LD_LIBRARY_PATH="$PYTHON_LIBDIR" cargo test --lib   # Linux
DYLD_LIBRARY_PATH="$PYTHON_LIBDIR" cargo test --lib # macOS
```

CI (`.github/workflows/bioregion-rs.yml`) does this automatically.
