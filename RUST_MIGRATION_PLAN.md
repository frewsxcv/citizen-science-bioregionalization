# Rust Migration Plan

A file-by-file plan for rewriting the bioregionalization pipeline in Rust, plus the
interop question: **can we pass DataFrames/LazyFrames across the Rust↔Python boundary?**

## TL;DR

- **Yes, DataFrames pass cleanly both directions.** Two complementary mechanisms:
  1. **`pyo3-polars`** — zero-copy `pl.DataFrame`/`pl.LazyFrame`/`pl.Series` across FFI
     via the Arrow C data interface. Build with `maturin` as a native module.
  2. **The parquet cache boundary you already have.** `cache_parquet()` already sinks
     every pipeline stage to parquet and rescans it. That is a ready-made seam: a Rust
     stage and a Python stage interoperate just by reading/writing the same parquet file.
- **A 100% rewrite is not the right target.** Two ML kernels have no faithful Rust
  equivalent today: **UMAP** (`umap-learn`) and **connectivity-constrained Ward
  agglomerative clustering** (`sklearn`). Everything else — geometry, H3, graph coloring,
  group-bys, Fisher's exact, PERMANOVA, cluster metrics, GeoJSON/JSON output — ports
  cleanly. Recommendation: a **hybrid** where Rust owns the deterministic compute and
  Python keeps notebook orchestration, all plotting, and (initially) the two hard kernels.
- Migrate **file-by-file behind the parquet boundary**, diffing Rust output against the
  current Python output at each stage before switching over.

---

## Current architecture (what we're porting)

The pipeline is a DAG of two node kinds, orchestrated by the `marimo` notebook
(`notebook.py`), with each stage cached to parquet:

- **"dataframes"** (`src/dataframes/*`): `dataframely`-validated Polars frames. Each has a
  `Schema` class + a `build_*` function. This is ~70% of the code and almost all of it is
  pure Polars + a few scientific calls.
- **"matrices"** (`src/matrices/*`): NumPy/SciPy distance & connectivity matrices that
  don't fit the tabular model.

Pipeline order (from `notebook.py`):

```
darwin_core → geocode → geocode_neighbors → taxonomy → geocode_taxa_counts
  → (filter top taxa) → geocode (filtered)
  → geocode_connectivity_matrix
  → geocode_distance_matrix         [RobustScaler → UMAP → braycurtis pdist]
  → geocode_cluster_multi_k         [AgglomerativeClustering ward+connectivity, per k]
  → geocode_cluster_metrics         [silhouette / CH / DB / inertia] → optimal_k (kneed)
  → geocode_cluster (single k)
  → cluster_taxa_statistics → cluster_neighbors → cluster_distance_matrix
  → cluster_color                   [greedy graph coloring | taxonomic UMAP+MDS]
  → cluster_boundary                [shapely union of hex boundaries]
  → cluster_significant_differences [Fisher's exact + scoring]
  → permanova_results               [skbio permutation test]
  → significant_taxa_images
  → geojson / json / html output + plots
```

---

## Interop: passing frames between Rust and Python

### Mechanism 1 — `pyo3-polars` (in-process, zero-copy)

Polars is itself a Rust library; the Python package is a thin wrapper over the same
`polars` crate. `pyo3-polars` exposes wrapper types (`PyDataFrame`, `PyLazyFrame`,
`PySeries`, `PyExpr`) that convert to/from the Rust `polars` types across the Arrow C
data interface — no serialization, no copy of the underlying buffers.

```rust
use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::{PyDataFrame, PyLazyFrame};

#[pyfunction]
fn build_geocode_taxa_counts(darwin_core: PyLazyFrame, precision: u8) -> PyResult<PyLazyFrame> {
    let lf: LazyFrame = darwin_core.into();
    let out = lf /* ... group_by / agg in Rust ... */;
    Ok(PyLazyFrame(out))
}

#[pymodule]
fn bioregion_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_geocode_taxa_counts, m)?)?;
    Ok(())
}
```

```python
# Python side, from the notebook:
import bioregion_rs
lf = bioregion_rs.build_geocode_taxa_counts(darwin_core_lf, precision=4)  # pl.LazyFrame in/out
```

Caveats worth knowing up front:
- **The Python-polars and Rust-`polars` versions must be ABI-compatible.** Pin them
  together and bump in lockstep. This is the one real operational tax of `pyo3-polars`.
- LazyFrames cross the boundary as **serialized logical plans**; `.collect()` still happens
  wherever you call it. Works, but plan-node compatibility is what the version pin protects.
- **Geometry columns** (`polars-st`) are stored as WKB `Binary`. That is plain bytes and
  crosses the FFI fine; on the Rust side decode with the `wkb`/`geo` crates.
- Build/packaging via **`maturin`**; add `bioregion_rs` as a dependency the notebook imports.

### Mechanism 2 — the parquet cache boundary (already present)

`src/cache_parquet.py` writes each stage to `{DATA_DIR}/polars_cache/{hash}.parquet` and
returns a fresh `scan_parquet`. Because every stage is already a parquet round-trip, you
can replace one stage's *producer* with Rust while every *consumer* stays in Python (or
vice versa) with zero code coupling — they only share a parquet file and a schema.

Use this as the **migration harness**: for each file, run the Python stage and the new Rust
stage against the same input parquet and assert the outputs match (exact for
deterministic stages; distributional/tolerance-based for the stochastic ML stages).

**Recommended split of the two mechanisms:** use the **parquet boundary** as the migration
scaffold and correctness gate (coarse-grained, one file at a time), and reach for
**`pyo3-polars`** once a cluster of Rust stages is stable and you want to fuse them
in-process to skip intermediate parquet I/O.

---

## Portability tiers (per dependency)

| Python dep | Used for | Rust path | Difficulty |
|---|---|---|---|
| `polars` | everything tabular | `polars` crate (same engine) | trivial |
| `polars-h3` / H3 | geocoding, grid_ring, boundary | `h3o` (pure-Rust H3) | easy |
| `polars-st` / `shapely` | points, polygons, intersect, union | `geo`, `geo-types`, `wkb`; `geos` if needed | moderate |
| `dataframely` | schema validation | custom validators over polars schema, or `pola-rs` checks | easy-moderate |
| XML `meta.xml` parse | Darwin Core archive | `quick-xml` | easy |
| `networkx` | greedy coloring, connected components | `petgraph` | moderate |
| `scipy.stats.fisher_exact` | significance test | port (hypergeometric); `statrs` for distributions | moderate |
| `scikit-bio` permanova | permutation test | port (it's a permutation F-test) | moderate |
| `kneed` | elbow selection | port Kneedle (small algorithm) | easy |
| sklearn silhouette/CH/DB | cluster metrics | port formulas; `linfa` partial | moderate |
| `RobustScaler` | feature scaling | port (median/IQR) | easy |
| `pdist(braycurtis)` / `squareform` | distance matrix | port (simple loops) | easy |
| **`umap-learn`** | dim reduction (2 places) | **no faithful equiv** (`annembed` is close-ish) | **hard / keep in Python** |
| **sklearn `AgglomerativeClustering` ward+connectivity** | clustering | **no equiv** (`kodama` = ward but no connectivity) | **hard / port or keep** |
| `sklearn.manifold.MDS` | taxonomic color layout | port SMACOF, or keep | hard-ish |
| `matplotlib`/`seaborn`/`altair`/`folium` | plots & maps | **keep in Python** | n/a |
| `marimo` | notebook orchestration | **keep in Python** (thin driver) | n/a |

---

## File-by-file plan

Ordered by phase. Each file lists what it does and the Rust approach. ✅ clean port,
⚠️ needs a ported algorithm, 🔴 hard kernel (keep in Python initially).

### Phase 0 — scaffolding (no behavior change) — ✅ DONE

Implemented in `bioregion_rs/` (see `bioregion_rs/README.md`).

- ✅ `bioregion_rs` crate created (`maturin` + `pyo3` 0.28 + `pyo3-polars` 0.27 +
  `h3o` 0.8), Rust `polars` pinned to **0.54.4** to match Python `polars` **1.37.0**.
  `maturin` added as a dev dependency of the root project; builds into the uv venv via
  `uv run maturin develop --manifest-path bioregion_rs/Cargo.toml`.
- ✅ **Diff harness** (`bioregion_rs/harness.py`) runs each Rust function against its
  Python counterpart on shared input and asserts equality. All checks pass:
  - plain PyO3 boundary — `darken_hex_color` matches `src/colors.py` across cases/factors;
  - pyo3-polars DataFrame boundary — `select_geocode` (h3o) matches
    `src/geocode.py::select_geocode_lf` (`polars-h3`) exactly at precisions 2/4/6/8;
  - parquet stage-handoff boundary round-trips.
  `h3o` produces **bit-identical** cell ids to `polars-h3` — the H3 port is drop-in.
- ✅ Two proof-of-concept ports landed (`darken_hex_color`, `select_geocode`) plus Rust
  unit tests (`cargo test --lib`).

**Findings that update the plan:**
- **The `lazy`/streaming feature set of polars 0.54.4 does not compile on the current
  nightly toolchain** (feature-unification bugs in `polars-ops` and `polars-stream`).
  Phase 0 uses the **eager DataFrame API only**. This is fine: the robust interop path is
  Arrow-FFI DataFrame transfer + the parquet boundary, both proven working. **LazyFrame
  plan-passing (`PyLazyFrame`) is deferred** — consistent with it being the
  version-sensitive path. Revisit when a polars/toolchain combo builds `lazy` cleanly, or
  pin a fixed polars patch.
- Implication for migration: stages hand off via **parquet** (or eager DataFrames across
  FFI), not lazy plans, until the lazy path is restored. Functions that are naturally lazy
  in Python (`build_*_lf`) get a `.collect()` at the Rust boundary for now.

Still TODO for the schema contract (carry into Phase 1): port each `dy.Schema` to a Rust
struct describing column names/dtypes/nullability + validation rules so both sides agree
on the parquet schema. Not blocking — done per-file as each `build_*` is ported.

### Phase 1 — leaves & pure tabular/IO (in progress)

Ported functions live in per-file Rust modules mirroring the Python layout
(`bioregion_rs/src/colors.rs`, `geocode.rs`, `dataframes/geocode.rs`, `wkb.rs`), each
verified against Python in `bioregion_rs/harness.py`.

- `src/colors.py` — ✅ `darken_hex_color` (Phase 0).
- `src/geocode.py` — ✅ DONE: `select_geocode`, `with_geocode`, `filter_by_bounding_box`
  (`h3o` + eager polars). Verified against `polars-h3` / Python across precisions.
- `src/dataframes/geocode.py` — ✅ DONE: `build_geocode` (H3 cell → center point +
  boundary polygon + edge-intersect test). `h3o` for center/boundary, hand-rolled WKB
  encoders (`wkb.rs`), `geo`'s `Intersects` for the bbox-boundary test. **Geometry matches
  Python bit-for-bit** (center coord error 0.0, boundary symmetric-difference area 0.0);
  `is_edge` matches exactly including True cases.
- `src/types.py`, `src/constants.py`, `src/defaults.py`, `src/logging.py`,
  `src/country_bbox.py` (static bbox data) — TODO, trivial ports (done as needed).
- `src/darwin_core_utils.py` — TODO: parquet/CSV scan + `meta.xml` parse + column renames.
  `quick-xml` for meta; polars scan in Rust. **Note:** this is lazy/streaming + IO-bound and
  the `lazy` feature is currently disabled (Phase 0 finding), so it stays in Python for now.
  (Also confirm the Rust polars build enables the cloud/object-store feature for `gs://`.)
- `src/dataframes/darwin_core.py` — TODO: schema + bbox filter + column select.
- `src/dataframes/taxonomy.py` — ✅ DONE: `build_taxonomy` (distinct
  (scientificName, gbifTaxonId) pairs restricted to known geocodes, each assigned a
  synthetic `taxonId`). Uses `DataFrame::group_by`/`unique`/`with_row_index` from
  **`polars-core`** directly — no `polars-ops`/joins/`is_in` needed (semi-join and
  dedup done as eager, hand-rolled filters, consistent with the no-`lazy` constraint).
  Verified against Python by comparing the (scientificName, gbifTaxonId) pair set and
  that `taxonId` is a 0..n bijection (row order — and thus which literal `taxonId` a
  pair gets — is not guaranteed to match Python's `.unique()` ordering, only the set
  and the bijection are).
- `src/dataframes/geocode_taxa_counts.py` — ✅ DONE: `build_geocode_taxa_counts` (the
  core `build_geocode_taxa_counts_lf` aggregation; `filter_top_taxa_lf` is deferred —
  it's an optional scaling filter, not on the critical path). The Python join against
  taxonomy (on scientificName+gbifTaxonId) and the group-by-sum are hand-rolled with a
  `HashMap` lookup + `polars-core`'s (deprecated but functional) `GroupBy::sum`, for the
  same reason as `taxonomy.rs` — no `polars-ops` dependency. Verified by resolving both
  engines' output back through their own taxonomy to (geocode, scientificName,
  gbifTaxonId, count) and comparing as sets (sidesteps the taxonId-ordering
  non-determinism noted above).
- `src/dataframes/geocode_neighbors.py` — ✅ DONE: `build_geocode_neighbors` and
  `build_geocode_neighbors_no_edges` (H3 `grid_ring_fast(1)` for direct adjacency; a
  hand-rolled union-find + brute-force nearest-point search for the connectivity fixup,
  instead of `petgraph` — the graph is tiny per bioregion and this avoids a new
  dependency). The public `graph()` helper (produces an `nx.Graph` for downstream
  consumers) stays in Python unchanged — it's a thin, Python-object-specific wrapper
  around whichever engine produced the `GeocodeNeighborsSchema` DataFrame. Verified
  against Python by comparing `direct_neighbors`/`direct_and_indirect_neighbors` as
  per-geocode sets (order-independent) and confirming both outputs are a single
  connected component; the indirect-edge tie-break (nearest pair when multiple
  candidates are equidistant) matched Python exactly on the test fixture, though this
  isn't guaranteed in general — see the caveat in `geocode_neighbors.rs`.
- `src/matrices/geocode_connectivity.py` — ✅ DONE: `build_geocode_connectivity_matrix`
  builds the dense 0/1 adjacency matrix from `direct_and_indirect_neighbors`, returned
  as a nested `Vec<Vec<i64>>` (plain Python list of lists — no `ndarray`/`sprs`
  dependency; PyO3 converts nested `Vec<u8>` to Python `bytes`, which is why cells are
  `i64` rather than a narrower type) for the Python side to wrap with `np.array(...)`.
  The single-connected-component invariant is checked via a small BFS (instead of
  `networkx`) and raises instead of Python's `assert`. Verified against
  `GeocodeConnectivityMatrix.build`'s output for exact matrix equality.

This closes out Phase 1's non-trivial files. `bioregion_rs` now has CI
(`.github/workflows/bioregion-rs.yml`): `cargo test --lib`, `maturin develop`, and
`harness.py` run on every push, so a Rust change that breaks parity with Python fails
CI instead of relying on someone running the harness locally. Note this is a
crate-level check only — the main pipeline's own CI (`run.yml`) still runs the
all-Python `notebook.py` end-to-end and does not yet invoke `bioregion_rs` at all,
since no pipeline call site has been switched over to Rust yet (see "Recommended end
state" above — that cutover is a later step, done file-by-file once each port is
validated).

### Phase 2 — graph, geometry, and stats (⚠️ ported algorithms)

- `src/dataframes/cluster_taxa_statistics.py` — ✅ DONE: `build_cluster_taxa_statistics`
  (per-taxon count/average, overall and per cluster). Same eager, hand-rolled
  join/group-by pattern as Phase 1 (`HashMap`s, no `polars-ops`): the join against
  `taxonomy_df` is really just a semi-filter (taxonId is a unique key there, so it
  never fans out rows and no taxonomy column is used downstream), and the join
  against the geocode→cluster mapping is a real inner join, implemented as a
  `HashMap<geocode, cluster>` lookup. Verified against Python by comparing
  `(cluster, taxonId, count, average)` as a row set, using a synthetic geocode→cluster
  assignment in the harness since real clustering (sklearn Ward) is Phase 3.
- `src/dataframes/cluster_neighbors.py` — ✅ DONE: `build_cluster_neighbors` derives
  which clusters are (direct/indirect) neighbors of each other from geocode-level
  adjacency plus a geocode→cluster mapping. Turned out not to need `petgraph` or any
  connected-components logic at all — it's a pure derivation (for each geocode's
  neighbors, note the pair of clusters if they differ), done with a
  `HashMap<geocode, cluster>` lookup. Mirrors Python's fail-fast behavior (errors
  instead of silently skipping) if a geocode is missing from the cluster mapping. The
  `to_graph()` helper (produces an `nx.Graph`) stays in Python, same reasoning as
  `geocode_neighbors.graph()`. Verified against Python by comparing neighbor sets per
  cluster, using a synthetic geocode→cluster assignment (Phase 3, out of scope here).
- `src/matrices/cluster_distance.py` — ✅ DONE: `build_cluster_distance_matrix`.
  Unlike `geocode_distance.py` (Phase 3), this file has **no UMAP step** — it's just
  `RobustScaler` (median/IQR per column, hand-ported: numpy's linear-interpolation
  percentile, IQR-of-0 leaves a column unscaled per sklearn's documented behavior) then
  `pdist(metric="braycurtis")`, both fully portable, so this is a complete port, not
  scaffolding for later. Returns a condensed (scipy `pdist`-order) distance vector plus
  the cluster IDs in row order, rather than a `ClusterDistanceMatrix` object — row/column
  order doesn't need to match Python's `pivot()` exactly, since both RobustScaler
  (per-column) and Bray-Curtis (order-invariant sums) give identical distances under any
  *consistent* column permutation. Verified against `ClusterDistanceMatrix.build` by
  looking up pairwise distances by cluster-ID pair (order-independent). **Caveat found
  while testing:** with exactly 2 clusters, RobustScaler always scales every non-constant
  column to a perfect ±1 pair, making Bray-Curtis's `sum(|u+v|)` denominator ~0 for every
  such column — the distance becomes a huge, floating-point-rounding-dominated number
  that two independent implementations have no reason to agree on bit-for-bit. This is a
  property of the algorithm at k=2, not a port bug; the harness test uses more clusters
  to avoid it, but real runs with `min_clusters=2` could hit this instability in Python
  today too.
- `src/dataframes/cluster_color.py` **(geographic path only)** — ✅ DONE:
  `build_cluster_color`. No `petgraph` needed — greedy coloring (`largest_first`
  ordering: nodes sorted by degree descending, stable-sorted so ties keep original
  order, then each node gets the smallest color index not used by an already-colored
  neighbor) is a ~20-line hand-rolled algorithm, exactly matching
  `networkx.coloring.greedy_color(G, strategy="largest_first")`. The palette step
  (`sns.color_palette("YlOrRd", n)`) turned out to be fully portable too: reverse-
  engineered matplotlib's exact sampling algorithm (`LinearSegmentedColormap`'s 9
  control points per RGB channel, piecewise-linear interpolation, `floor(x*256)`
  index quantization; seaborn samples at `linspace(0,1,n+2)[1:-1]`, excluding the
  colormap's extremes) and verified it reproduces `sns.color_palette("YlOrRd",
  n).as_hex()` bit-for-bit for n up to 15. `darken_hex_color` (Phase 0) is reused
  directly. (Taxonomic path uses UMAP+MDS — see Phase 3.)
- `src/dataframes/cluster_boundary.py` — ✅ DONE: `build_cluster_boundary`, using
  `geo::unary_union` (no `geos` bindings needed) over WKB-decoded hexagon boundaries.
  Added general WKB Polygon/MultiPolygon decode+encode to `wkb.rs` (rings + holes,
  not just the single-ring hexagons `dataframes/geocode.rs` produces) to round-trip
  through `geo::Polygon`/`MultiPolygon`. A single-geocode cluster is encoded as a bare
  `Polygon` (matching Python exactly); a multi-geocode cluster is always encoded as a
  `MultiPolygon` (`unary_union`'s return type), even when the pieces fully merge into
  one shape — unlike shapely, which collapses that case to a bare `Polygon`. This is a
  WKB type-tag difference only, not a shape difference (GEOS treats a one-polygon
  `MultiPolygon` as topologically `.equals()` the bare `Polygon`, confirmed while
  testing). **Geometry-robustness caveat found while verifying:** `geo`'s
  `i_overlay`-based union and GEOS's union are different implementations, so the
  unioned shapes differ by a tiny amount (~2e-7 relative, i.e. floating-point-level
  vertex differences at shared hexagon edges) — the harness compares with a relative
  symmetric-difference tolerance instead of exact equality, unlike the exact/bit-for-bit
  checks used everywhere else in this crate. This is the exact "geometry robustness"
  risk this plan flagged at the top; in practice `geo` was accurate enough that `geos`
  bindings weren't needed.
- `src/dataframes/cluster_significant_differences.py` — ✅ DONE:
  `build_cluster_significant_differences`. **No `statrs` needed** — Fisher's exact
  (`scipy.stats.fisher_exact(table, alternative="two-sided")`) is hand-ported from
  scipy's actual C-level 2x2 algorithm (mode + binary search over the hypergeometric
  tail, `epsilon=1e-14` tie tolerance — read straight from scipy's source, since its
  docstring only describes the *result*, not how it's computed), backed by a
  from-scratch Lanczos `ln_gamma` (only ever called with argument >= 1 here, so the
  reflection formula for x < 0.5 isn't needed). Verified bit-for-bit (< 1e-12) against
  both worked examples in scipy's `fisher_exact` docstring. The scoring/normalization
  math (linear log2-fold-change scaling, logarithmic count scaling, the two composite
  scores) is hand-rolled arithmetic over plain Rust structs rather than polars
  expressions, consistent with this crate's eager/hand-rolled style. Verified against
  Python on a small hand-built (not sample-archive-derived) fixture designed to
  actually trigger the significance path — the sample archive's real counts are all
  well under `MIN_COUNT_THRESHOLD=5`.
- `src/dataframes/permanova_results.py` — ✅ DONE: `build_permanova_results`. The
  pseudo-F statistic is a deterministic function of the distance matrix + grouping
  (`s_T`/`s_W`/`F`, read from skbio's actual `_cutils.pyx` Cython source) and is
  ported exactly — verified bit-for-bit against `skbio.stats.distance.permanova`
  called directly with `permutations=0`. **The p-value cannot be made
  bit-reproducible**: `build_permanova_results_df` calls `permanova(..., seed=None)`,
  i.e. skbio's own Monte Carlo permutation test uses an *unseeded* RNG, so the
  p-value already differs between two separate Python runs, before Rust is even in
  the picture — "seed the RNG for reproducibility" (this plan's original suggestion)
  wouldn't fix that, since the call site never passes a seed. Ported the same
  algorithm (shuffle labels, recompute F, `p = (1+count(F_perm>=F_obs))/(1+permutations)`)
  with Rust's own `rand`-crate shuffle, which is the best achievable "equivalence" for
  an inherently randomized test — verified via a hand-built fixture: exact match on
  `test_statistic` (`permutations=0`), then a statistical-consistency check (both
  p-values within a few standard errors of each other) at `permutations=999`.
  **New dependency**: `rand = "0.9"` — already fully resolved transitively (`polars`
  depends on the same version), so this adds no real new dependency surface, unlike
  a from-scratch shuffle which risked getting the RNG quality wrong for a statistical
  test where that actually matters.
- `src/dataframes/geocode_cluster_metrics.py` — ✅ DONE: `build_geocode_cluster_metrics`
  + `select_optimal_k_elbow`. Silhouette (mean), Calinski-Harabasz, and Davies-Bouldin
  formulas read directly from sklearn's `_unsupervised.py` source (not just its
  docstrings) — note Calinski-Harabasz/Davies-Bouldin use `X=dm_square` with no
  `metric="precomputed"`, i.e. the existing Python code treats each row of the square
  distance matrix as an n-dimensional "feature vector" and computes ordinary Euclidean
  distances between rows; this port preserves that (unusual but pre-existing)
  methodology rather than "fixing" it. Inertia is the codebase's own hand-rolled
  distance-matrix formula (not an sklearn function). Elbow detection uses the **`kneed`
  crate** rather than a hand-rolled Kneedle port. A first pass hand-rolled it
  (specialized to this call site's fixed `curve="convex"`, `direction="decreasing"`,
  `interp_method="interp1d"` parameters) and *looked* correct — it matched on every
  test case tried, including sklearn/`kneed` reference values — but excluded the
  curve's first/last point from local-extrema consideration, which silently diverges
  from scipy's actual `argrelextrema(mode="clip")` semantics whenever the first point
  isn't the strict global extreme (a real possibility: clustering metrics aren't
  guaranteed monotonic in k). Concretely: `y=[90,100,50,20,18,17]` gave elbow=5
  by hand vs. Python `kneed`'s real elbow=2. Caught by deliberately constructing a
  non-monotonic curve to stress-test the boundary case, *after* the hand-rolled
  version had already merged — see the regression test
  `elbow_matches_python_kneed_on_non_monotonic_curve`. Swapped to the `kneed` crate
  (crates.io, v1.0, an independent Rust port of the same Python `kneed` package,
  correctly handles this case) rather than fixing the hand-rolled version a second
  time — not worth re-deriving `argrelextrema`'s boundary semantics by hand twice.
  **Trade-off worth knowing**: `kneed` pulls in a real dependency tree (`nalgebra`,
  ~18 pinned `glam` versions, `polyfit-rs`, `anyhow`) for its polynomial-interpolation
  path, which we don't use (`interp_method="interp1d"` only) — this is the one place
  in the crate so far where "use an existing library" clearly won on correctness risk
  but cost noticeably more in dependency surface than the hand-rolled alternatives
  used everywhere else (Fisher's exact, `YlOrRd`, robust scaling, etc.). `get_elbow_analysis`/
  `get_metrics_summary`/`get_metric_interpretations` stay in Python (plotting/presentation
  helpers).
- `src/dataframes/geocode_silhouette_score.py` — TODO: per-sample silhouette (needs
  `silhouette_samples`, not just the mean `silhouette_score` this PR ported for
  `geocode_cluster_metrics.py` — same underlying formula, extended to return one score
  per point instead of the average).
- `src/cluster_optimization.py` — thin orchestration over metrics + elbow. ✅
- `src/dataframes/significant_taxa_images.py` — image URL/id lookup join. ✅ (confirm no
  network fetch; if it hits an API, keep that thin bit in Python or use `reqwest`).
- `src/geojson.py`, `src/output.py`, `src/render.py` — GeoJSON + JSON (+ HTML) emit.
  `geojson`/`serde_json` crates; decode WKB with `wkb`+`geo`. ✅

### Phase 3 — hard ML kernels (🔴 keep in Python until validated)

- `src/matrices/geocode_distance.py` — three sub-steps:
  - `RobustScaler` → ✅ port (median/IQR per column).
  - `pdist(braycurtis)` / `squareform` → ✅ port.
  - **UMAP** (`umap.UMAP(metric="braycurtis")`) → 🔴 no faithful Rust equivalent. Keep in
    Python (call it via `pyo3` back into Python, or leave this stage Python-side and pass
    the reduced-feature parquet to Rust). `annembed` exists but won't reproduce sklearn/UMAP
    output — a swap here changes clustering results and needs its own validation.
- `src/dataframes/geocode_cluster.py` — **connectivity-constrained Ward agglomerative
  clustering** per k. 🔴 `kodama` does Ward but **not** connectivity constraints; sklearn's
  `ward_tree` with a connectivity graph would need a direct port (feasible — it's a
  heap-based nearest-merge over the connectivity structure — but it's the single most
  involved algorithm here). Keep in Python first; port behind the parquet boundary and diff
  labels against Python before switching.
- `src/dataframes/cluster_color.py` **(taxonomic path)** — UMAP(`metric="precomputed"`) +
  HSV mapping, and `MDS` usage. 🔴 depends on UMAP/MDS. The **geographic** default path is
  already Rust-ready (Phase 2); only migrate taxonomic coloring after UMAP is resolved.

### Phase 4 — stays in Python (the right boundary)

- `src/plot/*` (matplotlib/seaborn/altair), map rendering (folium) — Rust plotting is weak
  and these consume already-computed frames. Keep in Python; they read Rust-produced parquet.
- `notebook.py` (`marimo`) — orchestration/CLI/report. Keep as a thin Python driver that
  calls `bioregion_rs` functions and passes frames between stages.

---

## Recommended end state

```
┌────────────────────────── Python ──────────────────────────┐
│  marimo notebook (CLI, orchestration, report)               │
│  plotting: matplotlib / seaborn / altair / folium           │
│  hard ML kernels: UMAP, Ward+connectivity clustering        │
│      (until Rust ports are validated)                       │
└───────────────▲───────────────────────────▲────────────────┘
                │ pyo3-polars (frames)       │ parquet cache
                │ zero-copy in-process       │ (stage boundary)
┌───────────────┴───────────────────────────┴────────────────┐
│                        bioregion_rs (Rust)                  │
│  polars · h3o · geo/wkb · petgraph · statrs                 │
│  darwin_core · geocode · neighbors · taxa_counts ·          │
│  taxa_statistics · significant_differences · permanova ·    │
│  cluster_metrics · silhouette · boundaries · colors(geo) ·  │
│  connectivity/distance matrices · geojson/json output       │
└─────────────────────────────────────────────────────────────┘
```

## Risks / open questions

1. **`pyo3-polars` version lockstep** — the main ongoing maintenance cost. Pin polars on
   both sides and upgrade together.
2. **UMAP** determinism/equivalence — the biggest blocker to a full rewrite. If exact
   parity matters, UMAP likely stays Python permanently. If the pipeline can tolerate a
   different-but-valid embedding, evaluate `annembed` on cluster-quality metrics, not on
   output equality.
3. **Ward + connectivity clustering** — portable but the most work; port it early behind the
   parquet diff harness so you can verify label agreement against sklearn before committing.
4. **Geometry robustness** — `shapely`/GEOS handle degenerate unions gracefully; pure-Rust
   `geo` boolean ops are improving but for the boundary-union step consider `geos` bindings
   if you hit robustness issues.
5. **Cloud parquet (`gs://`)** — ensure the Rust polars build enables the object-store/cloud
   feature so `darwin_core_utils` can read GBIF snapshots directly.
6. **`dataframely` validation semantics** — decide how faithfully to reproduce its rule
   engine vs. a lighter custom validator; it affects how much of the schema layer you port.
```