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
- **A 100% rewrite is not the right target.** One ML kernel has no faithful Rust
  equivalent today: **UMAP** (`umap-learn`). The other hard kernel —
  **connectivity-constrained Ward agglomerative clustering** (`sklearn`) — turned out
  to have no library equivalent but *was* portable by reproducing sklearn's own
  `ward_tree` + `_hc_cut` algorithm directly, and is now cut over (see Phase 3).
  Everything else — geometry, H3, graph coloring, group-bys, Fisher's exact, PERMANOVA,
  cluster metrics, GeoJSON/JSON output — ports cleanly. Recommendation: a **hybrid**
  where Rust owns the deterministic compute and Python keeps notebook orchestration,
  all plotting, and (for now) UMAP.
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
| **sklearn `AgglomerativeClustering` ward+connectivity** | clustering | **ported** — reproduce sklearn's `ward_tree`+`_hc_cut` (no crate does connectivity-constrained Ward) | done (was: hard) |
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
- **The `lazy` feature compiles fine** on the current stable toolchain (rustc 1.93.1)
  when `temporal` is enabled alongside it. The original Phase 0 note ("lazy doesn't
  compile") was recorded on a nightly toolchain and was really a narrow polars
  feature-unification bug, **not** a fundamental lazy limitation: enabling `strings`
  (which the pipeline needs, for `scientificName` etc.) without `temporal` leaves
  `polars-expr`'s string dispatch (`polars-expr-0.54.4/src/dispatch/strings.rs:13`,
  `use polars_time::prelude::StringMethods`) referencing the unlinked `polars_time`
  crate. Enabling `temporal` too resolves it; verified `bioregion_rs` `cargo check`s
  cleanly with `features = ["lazy", "temporal"]`.
- **We still use the eager DataFrame API only** — but for a different reason than
  originally stated. It's not that lazy can't build; it's that lazy interop buys
  nothing here. The pure-Polars stages (`taxonomy`, `geocode_taxa_counts`, the geocode
  filters) already run in the Rust engine, so passing `PyLazyFrame` plans across the
  boundary would execute identically (same memory, same speed). LazyFrame plan-passing
  is deferred as **unnecessary**, not blocked. The robust interop path is Arrow-FFI
  DataFrame transfer + the parquet boundary, both proven working.
- Implication for migration: stages hand off via **parquet** (or eager DataFrames across
  FFI), not lazy plans. Functions that are naturally lazy in Python (`build_*_lf`) get a
  `.collect()` at the Rust boundary.

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
- `src/darwin_core_utils.py` — ✅ PARTIAL: `meta.xml` parsing ported to Rust
  (`parse_darwin_core_meta`, `quick-xml`); `_parse_meta` now delegates to it and
  reconstructs the `_Meta` dataclass. Verified against the sample archive
  (`test/test_darwin_core_utils.py::TestParseMeta`). **The scan itself stays in
  Python** (`scan_darwin_core_archive` / `build_darwin_core_raw_lf`): it is pure
  Polars plan construction that must run in the pipeline's own Polars engine.
  Returning the scan as a `PyLazyFrame` was investigated and rejected — pyo3-polars
  serializes the logical plan and its `DSL_SCHEMA_HASH` only matches when the Rust
  crate is built from the *exact same commit* as the installed pip polars wheel
  (not merely the same version: crates.io `0.54.4` and pip `1.42.0` — both nominal
  workspace 0.54.4 — have different DSL hashes). That lockstep coupling isn't worth
  it for zero runtime gain, since the scan runs in the same Polars engine either way.
- `src/dataframes/darwin_core.py` — TODO: schema + bbox filter + column select.
- `src/dataframes/taxonomy.py` — Rust port ✅ exists (`build_taxonomy`: distinct
  (scientificName, gbifTaxonId) pairs restricted to known geocodes, each assigned a
  synthetic `taxonId`; uses `DataFrame::group_by`/`unique`/`with_row_index` from
  **`polars-core`** directly — no `polars-ops`/joins/`is_in` needed — and is verified
  correct against Python via `harness.py`), but **🔴 the real pipeline's
  `build_taxonomy_lf` was NOT cut over to call it** — reverted after 3/3 CI failures on
  the full GBIF-snapshot job (`gh pr #27`). Root cause: cutting over meant collecting
  `darwin_core_lf`'s selected columns (including `scientificName`, a string column)
  into an eager `DataFrame` *before* calling Rust, so Rust could do the bbox-filter +
  geocode + semi-join + dedup itself. But the *original* Python implementation never
  collects at all — it stays a `pl.LazyFrame` end to end (`TaxonomySchema.validate(lf,
  eager=False)`), letting Polars' lazy engine push the semi-join and `.unique()` dedup
  down *before* materializing anything. Since the real output is one row per distinct
  species (thousands) but the raw input is one row per occurrence in the bbox
  (potentially tens of millions for a large snapshot), materializing the *raw* rows
  before dedup — as the cutover did — is orders of magnitude more memory than
  materializing the *deduped* result, as the original lazy pipeline does. This OOM'd
  the CI runner (`gh run view`'s generic "runner lost communication with the server",
  reproduced 3/3 times, ~56-58min in every time) even though it passed fine against the
  small `test/sample-archive/` fixture used for local verification — the bug only shows
  up at real-snapshot scale. **Lesson for any future cutover of this shape:** a
  "collect the raw input, then reduce inside Rust" boundary only works when the
  Rust-side reduction is *cheap relative to the raw input size*; when the real value is
  in reducing cardinality by orders of magnitude (dedup/aggregation over a huge raw
  table), that reduction has to happen before crossing the Python/Rust boundary, not
  after — which the current PyO3/`pyo3-polars` interop (whole-`DataFrame`-at-a-time,
  no chunked/streaming API) can't do without keeping the reduction in lazy Polars.
- `src/dataframes/geocode_taxa_counts.py` — same status and same root cause as
  `taxonomy.py` above: Rust port ✅ exists (`build_geocode_taxa_counts`, verified via
  harness.py; `filter_top_taxa_lf` was always deferred, not on the critical path), but
  **🔴 `build_geocode_taxa_counts_lf` was NOT cut over** — reverted after 3/3 CI failures
  on the full GBIF-snapshot job (`gh pr #28`, same "runner lost communication" pattern,
  ~1h-1h4min elapsed every time). It has the same shape: the real output is aggregated
  (geocode, taxonId) counts, but the cutover collected raw per-occurrence rows
  (including `scientificName`) before aggregating in Rust, instead of letting Polars'
  lazy `group_by` aggregate before materializing.
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
(`.github/workflows/bioregion-rs.yml`): `cargo test --lib`, `maturin develop`
(later: `uv sync`, once it became a workspace dependency — see Phase 4), and
`harness.py` run on every push, so a Rust change that breaks parity with Python fails
CI instead of relying on someone running the harness locally. At this point in the
project this was still a crate-level check only — the main pipeline's own CI
(`run.yml`) ran the all-Python `notebook.py` end-to-end and didn't invoke
`bioregion_rs` at all, since no pipeline call site had been switched over to Rust yet
(that cutover work is Phase 4, below).

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
- `src/dataframes/geocode_silhouette_score.py` — ✅ DONE: `build_geocode_silhouette_score`.
  Same per-point formula as `geocode_cluster_metrics.py`'s mean silhouette score (both
  read from sklearn's `_silhouette_reduce`), extended to emit one score per geocode
  instead of just the average — plus a leading `geocode = null` row per k holding that
  average, matching the Python output shape exactly. Verified against
  `build_geocode_silhouette_score_df` on the same hand-built multi-k fixture used for
  `geocode_cluster_metrics.py`.
- `src/cluster_optimization.py` — ✅ DONE: `optimize_num_clusters`. Genuinely thin
  orchestration, as the plan predicted: calls the already-ported
  `build_geocode_cluster_metrics` and `find_elbow_point` directly (in-process, no
  Python round-trip), falling back to the highest `combined_score` if no elbow point
  is found. No new math. Verified against Python's `optimize_num_clusters` on the same
  multi-k fixture used for `geocode_cluster_metrics.py`, deliberately using only 2
  distinct k values so the test exercises the "no elbow found" fallback path (elbow-found
  behavior itself is already covered by `geocode_cluster_metrics.py`'s own tests).
- `src/dataframes/significant_taxa_images.py` — 🔴 STAYS IN PYTHON, confirmed:
  `_fetch_wikidata_images` does a live SPARQL-over-HTTP POST to Wikidata. Checked
  whether a Rust SPARQL client would change that (a user-authored crate,
  [rust-sparqling](https://github.com/observ-ing/rust-sparqling), depends on
  `reqwest` + `tokio`) — it would work for the fetch itself, but it'd be the single
  heaviest dependency addition in this migration (bigger than `kneed`'s
  `nalgebra`/`glam` tree) and the *first* async code in a crate that's otherwise
  entirely synchronous PyO3, all to accelerate a one-shot, non-performance-critical
  network call whose actual "compute" (a dict lookup + join) is trivial. Not worth it;
  matches this plan's original call ("if it hits an API, keep that thin bit in
  Python").
- `src/geojson.py` — ✅ DONE: `build_geojson_feature_collection`, using the **`geojson`
  crate** (its `geo-types` feature converts our existing `geo::Polygon`/`MultiPolygon`
  directly; its pinned `geo-types` range unifies with the version `geo` already pulls
  in, so no duplicate). A first pass hand-built the JSON via `format!` to avoid a
  dependency — that version had a real bug (a copy-paste swap of the `color`/
  `fillColor` property values), caught only because a unit test happened to assert the
  exact JSON shape. Building typed `Feature`/`FeatureCollection` values and letting the
  crate serialize them removes that whole class of mistake, so the crate won out
  here despite the added dependency (unlike, say, `cluster_boundary.py`'s hand-rolled
  WKB encoding, which has no such structural-mistake risk). Still returns the
  FeatureCollection as a JSON string rather than a `geojson.FeatureCollection` Python
  object — nothing in the codebase does an `isinstance` check against that type (it's
  only ever handed to `geojson.dump`, which just needs something JSON-serializable).
  Required extending `wkb.rs` to decode a WKB *MultiPolygon* (not just `Polygon`),
  since `cluster_boundary.rs`'s output can be either shape depending on cluster size;
  the refactor also had to change `decode_polygon` to report how many bytes it
  consumed, so a MultiPolygon's embedded per-polygon WKB buffers can be walked in
  sequence. `write_geojson` (a plain file write) stays in Python — I/O, not compute.
  ✅ **Cut over** (Phase 4): `build_geojson_feature_collection` now calls this, parsing
  the returned JSON string back into a `geojson.FeatureCollection` via `geojson.loads`.
  Verified against
  `build_geojson_feature_collection` in `harness.py` on real boundaries derived from
  the sample archive (a mix of single-geocode `Polygon` and multi-geocode
  `MultiPolygon` clusters, exercising both shapes): properties compared exactly,
  geometry compared with the same relative-tolerance check as
  `cluster_boundary.py` (same underlying `geo`-vs-GEOS union discrepancy).
- `src/output.py` — ✅ DONE (`build_json_output` in `output.rs`, mirrors
  `write_json_output`'s assembly logic). Joins `cluster_boundary_df` with
  `cluster_color_df` (inner), and per cluster, `cluster_significant_differences_df`
  with `taxonomy_df` (inner) and `significant_taxa_images_df` (left) — a taxonId
  missing from `taxonomy_df` is dropped, one missing from
  `significant_taxa_images_df` gets a null `image_url`, matching Python's join
  semantics exactly. Boundary WKB is converted to a GeoJSON geometry via the same
  `wkb::decode_geometry` + `geojson` crate path as `geojson.rs`. Returns the
  assembled JSON as a `String` rather than writing it — `prepare_file_path` +
  the actual file write stay in Python (I/O, not compute, same rationale as
  `geojson.rs`'s `write_geojson`). ✅ **Cut over** (Phase 4): `write_json_output` now
  writes this string directly to disk instead of building the JSON by hand. Verified in
  `harness.py` against a hand-built fixture (reusing
  `test_build_cluster_significant_differences`'s data) that deliberately
  exercises both join edge cases (a dropped taxonId, a null `image_url`);
  `significant_taxa` lists are compared as order-independent sets since neither
  implementation's join guarantees row order.
- `src/render.py` (`features_to_polars_df`) — 🔴 NOT PORTED, likely dead code:
  confirmed via repo-wide grep that this function is referenced only by its own
  test (`test/test_render.py`), not by `notebook.py` or any other pipeline file.
  Not worth porting unless something starts calling it.

### Phase 4 — pipeline cutover (Rust ports → actually called by `notebook.py`)

Everything above this point was purely additive: `bioregion_rs` functions existed and
were verified correct against Python via `harness.py`, but nothing in the real
pipeline (`notebook.py` or any `src/*.py` module) ever imported `bioregion_rs`. This
phase made the pipeline actually call the Rust implementations, file by file.

**Packaging/CI prerequisite:** `bioregion_rs` was not a runtime dependency of the root
project at all — it was only built via a standalone `maturin develop` step in its own
CI job. Made it a `uv` workspace member (root `pyproject.toml`'s
`[tool.uv.workspace]`/`[tool.uv.sources]`, declared as `bioregion-rs`), added a Rust
toolchain + cargo cache to `run.yml`/`unittest.yml` (they'd never built the extension
before), and added a hand-maintained `bioregion_rs/bioregion_rs.pyi` stub so pyright
can typecheck call sites (PyO3 extensions don't auto-generate stubs; maturin bundles a
top-level `<module_name>.pyi` into the wheel automatically). Also found and worked
around a real footgun: `bioregion_rs/target/wheels/` can hold a stale pre-built wheel
that maturin's build backend reuses without reinvoking `cargo` if that directory
persists across builds — not reachable in CI (clean checkout every run), but the cache
steps deliberately exclude that directory to be safe regardless. See
`bioregion_rs/README.md`'s "Packaging" section for details.

**Cut over** (each function's public signature and `dataframely`-validated return type
kept identical, so `notebook.py` and every existing test needed zero changes; only the
function body's internals changed to delegate to `bioregion_rs`):
`src/dataframes/geocode.py` (`build_geocode_lf`), `src/dataframes/geocode_neighbors.py`
(both build functions), `src/matrices/geocode_connectivity.py`
(`GeocodeConnectivityMatrix.build`), `src/dataframes/cluster_taxa_statistics.py`,
`src/dataframes/cluster_neighbors.py`, `src/matrices/cluster_distance.py`
(`ClusterDistanceMatrix.build`), `src/dataframes/cluster_color.py` (geographic path
only), `src/dataframes/cluster_boundary.py`,
`src/dataframes/cluster_significant_differences.py`,
`src/dataframes/permanova_results.py`, `src/dataframes/geocode_cluster_metrics.py`
(`build_geocode_cluster_metrics_df` + `select_optimal_k_elbow`),
`src/dataframes/geocode_silhouette_score.py` (**plus** wiring `notebook.py`'s
silhouette cell to actually call it — it was dead code from the pipeline's
perspective before this, duplicating the same logic inline via sklearn instead),
`src/geojson.py`, `src/output.py`. `src/cluster_optimization.py` needed no code
changes — it's pure orchestration over two of the functions above, so it benefited
automatically once they were individually cut over.

A recurring theme during this phase: several now-Python-side helper functions became
dead code once their caller's internals moved to Rust (e.g. `add_cluster_column`,
`to_graph`, `iter_cluster_ids`, `_add_normalized_scores`, `build_geojson_feature`,
`_wkb_to_geojson`) and were deleted outright, while others stayed even though their
caller no longer used them internally, because a test file imports and exercises them
*directly* (e.g. `_reduce_connected_components_to_one`/`_df_to_graph` in
`geocode_neighbors.py`, `build_X`/`pivot_taxon_counts_for_clusters` in
`cluster_distance.py`, `_compute_inertia`/`_find_elbow_point` in
`geocode_cluster_metrics.py`) — those are left as-is, still under direct test, just no
longer on the hot path.

**Not cut over — reverted after real production failures:**
`src/dataframes/taxonomy.py` (`build_taxonomy_lf`) and
`src/dataframes/geocode_taxa_counts.py` (`build_geocode_taxa_counts_lf`). Both Rust
ports exist and are verified correct via `harness.py`, but wiring them into the real
pipeline caused the CI `gbif-snapshot` job (the full ~300M-row public GBIF dataset) to
OOM the runner 3/3 times (GitHub's generic "runner lost communication with the
server", ~56min-1h4min elapsed every time — never reproduced against the small
`test/sample-archive/` fixture used for local verification, which is why this wasn't
caught until real-scale CI). Root cause: both cutovers required collecting
`darwin_core_lf`'s selected columns (including `scientificName`, a string column) into
an eager `DataFrame` *before* calling Rust, so Rust could do the
filter/geocode/join/dedup-or-aggregate itself. But the *original* Python
implementations never collect early — they stay `pl.LazyFrame` end to end
(`eager=False`), letting Polars' lazy engine push the semi-join, `.unique()` dedup, and
`group_by` aggregation down *before* materializing anything. Since the real output for
both is orders of magnitude smaller than the raw input (one row per distinct species,
or one row per (geocode, taxonId) pair, vs. one row per raw occurrence — potentially
tens of millions at full-snapshot scale), materializing the *raw* rows before
reduction — which is what the cutover did — is orders of magnitude more memory than
materializing the *reduced* result, which is what the original lazy pipeline does.
**Lesson for any future cutover of this shape:** a "collect the raw input, then reduce
inside Rust" boundary only works when the Rust-side reduction is cheap relative to the
raw input size. When the real value is in reducing cardinality by orders of magnitude
(dedup/aggregation over a huge raw table), that reduction has to happen *before*
crossing the Python/Rust boundary — which the current PyO3/`pyo3-polars` interop
(whole-`DataFrame`-at-a-time, no chunked/streaming API) can't do without keeping the
reduction itself in lazy Polars. A real fix would mean either giving `bioregion_rs` a
streaming/chunked ingestion API, or restructuring these two functions so the
cardinality-reducing step (semi-join + dedup / group-by) happens lazily in Polars
first and only the cheap remainder (taxonId assignment / final formatting) goes
through Rust — out of scope for now. `build_taxonomy_lf`/`build_geocode_taxa_counts_lf`
remain pure Python; `bioregion_rs.build_taxonomy`/`build_geocode_taxa_counts` stay in
the crate, verified via `harness.py`, available for a future attempt at this.

### Phase 3 — hard ML kernels (🔴 keep in Python until validated)

- `src/matrices/geocode_distance.py` — three sub-steps:
  - `RobustScaler` → ✅ port (median/IQR per column).
  - `pdist(braycurtis)` / `squareform` → ✅ port.
  - **UMAP** (`umap.UMAP(metric="braycurtis")`) → 🔴 no faithful Rust equivalent. Keep in
    Python (call it via `pyo3` back into Python, or leave this stage Python-side and pass
    the reduced-feature parquet to Rust). `annembed` exists but won't reproduce sklearn/UMAP
    output — a swap here changes clustering results and needs its own validation.
- `src/dataframes/geocode_cluster.py` — **connectivity-constrained Ward agglomerative
  clustering** per k. ✅ DONE and ✅ **cut over** (Phase 4-style, in
  `dataframes/geocode_cluster.rs`: `build_geocode_cluster_multi_k`). `kodama` does Ward
  but **not** connectivity constraints, and no other crate does either, so this
  reproduces sklearn's own algorithm read straight from source:
  `sklearn.cluster._agglomerative.ward_tree` (the connectivity-constrained heap-based
  nearest-merge) + `_hc_cut` (cutting the full tree at each k) + the `compute_ward_dist`
  and `_get_parents` Cython helpers. Two properties made an *exact* deterministic port
  possible: (1) the pipeline passes the squareform distance matrix as `X` with the
  default `metric="euclidean"` (not `precomputed`), so each row is a feature vector —
  the same pre-existing methodology `geocode_cluster_metrics.rs` already preserves; and
  (2) sklearn's merge heap is keyed on `(inertia, row, col)` with every pair distinct,
  a strict total order with no ties, so the merge sequence is uniquely determined and a
  plain Rust `BinaryHeap` reproduces `heapq`'s pop order bit-for-bit. The one
  heap-array-order-sensitive step (`_hc_cut`'s label numbering) reimplements CPython
  `heapq`'s sift operations exactly, so the port reproduces sklearn's integer cluster
  labels, not merely an equivalent partition. Building the full tree **once** and cutting
  at every k (`compute_full_tree` is always true here — k is far below
  `max(100, 0.02·n_samples)`) also replaces the Python loop's per-k re-fit. Verified in
  `harness.py` against sklearn `AgglomerativeClustering(linkage="ward",
  connectivity=...)` on a connected 4×5 grid graph with *integer* coordinates
  (deliberately tie-heavy, to stress the merge tie-breaking): cluster labels match
  **exactly** for every k in 2..6. UMAP still runs in Python and produces the distance
  matrix this stage consumes — the boundary is `distance_matrix.condensed()`, a clean
  handoff.
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
│  hard ML kernel: UMAP (no faithful Rust equivalent)         │
└───────────────▲───────────────────────────▲────────────────┘
                │ pyo3-polars (frames)       │ parquet cache
                │ zero-copy in-process       │ (stage boundary)
┌───────────────┴───────────────────────────┴────────────────┐
│                        bioregion_rs (Rust)                  │
│  polars · h3o · geo/wkb · petgraph · statrs                 │
│  darwin_core · geocode · neighbors · taxa_counts ·          │
│  taxa_statistics · significant_differences · permanova ·    │
│  cluster_metrics · silhouette · ward+connectivity clustering│
│  boundaries · colors(geo) ·                                 │
│  connectivity/distance matrices · geojson/json output       │
└─────────────────────────────────────────────────────────────┘
```

**Status vs. this diagram (see Phase 4 above):** everything listed is ported and
verified correct in isolation; everything except `taxa_counts` and (upstream of it)
`taxonomy` is also cut over into the real pipeline. Those two remain pure Python —
cutting them over OOM'd the full-GBIF-snapshot CI job, since their value is in Polars'
lazy engine reducing a huge raw table *before* materializing, which the current
collect-then-call-Rust boundary can't preserve. UMAP is now the **only** hard ML
kernel still in Python; the Ward+connectivity clustering it feeds is cut over, taking
`distance_matrix.condensed()` as its clean handoff boundary.

## Risks / open questions

1. **`pyo3-polars` version lockstep** — the main ongoing maintenance cost. Pin polars on
   both sides and upgrade together.
2. **UMAP** determinism/equivalence — the biggest blocker to a full rewrite. If exact
   parity matters, UMAP likely stays Python permanently. If the pipeline can tolerate a
   different-but-valid embedding, evaluate `annembed` on cluster-quality metrics, not on
   output equality.
3. **Ward + connectivity clustering** — ✅ resolved. Ported by reproducing sklearn's own
   `ward_tree`+`_hc_cut` (no crate does connectivity-constrained Ward) and cut over; the
   harness verifies exact label agreement against sklearn, including on a deliberately
   tie-heavy fixture. This was the most involved single port but is fully deterministic.
4. **Geometry robustness** — `shapely`/GEOS handle degenerate unions gracefully; pure-Rust
   `geo` boolean ops are improving but for the boundary-union step consider `geos` bindings
   if you hit robustness issues.
5. **Cloud parquet (`gs://`)** — ensure the Rust polars build enables the object-store/cloud
   feature so `darwin_core_utils` can read GBIF snapshots directly.
6. **`dataframely` validation semantics** — decide how faithfully to reproduce its rule
   engine vs. a lighter custom validator; it affects how much of the schema layer you port.
```