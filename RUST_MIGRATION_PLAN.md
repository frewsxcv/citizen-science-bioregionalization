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
- `src/matrices/geocode_connectivity.py` — TODO: build sparse adjacency from neighbor lists
  (CSR via `sprs` or dense `ndarray`).

### Phase 2 — graph, geometry, and stats (⚠️ ported algorithms)

- `src/dataframes/cluster_taxa_statistics.py` — per-cluster taxon aggregation. ✅
- `src/dataframes/cluster_neighbors.py` — build cluster adjacency graph, connected
  components / neighbor expansion. `petgraph`. moderate
- `src/matrices/cluster_distance.py` — braycurtis `pdist` over cluster taxon vectors.
  Simple port. ✅
- `src/dataframes/cluster_color.py` **(geographic path only)** — `nx.greedy_color` →
  `petgraph`'s greedy coloring; palette mapping is deterministic. ✅
  (Taxonomic path uses UMAP+MDS — see Phase 3.)
- `src/dataframes/cluster_boundary.py` — union hex boundary polygons per cluster.
  `geo`'s `unary_union` / boolean ops (or `geos` bindings for robustness). moderate
- `src/dataframes/cluster_significant_differences.py` — 2×2 Fisher's exact per (cluster,
  taxon) + log2 fold change + normalized scoring. Port `fisher_exact` (hypergeometric tail
  sum; use `statrs`). The scoring math is plain polars. moderate
- `src/dataframes/permanova_results.py` — PERMANOVA is a permutation pseudo-F test over the
  condensed distance matrix + grouping. Fully portable; seed the RNG for reproducibility. moderate
- `src/dataframes/geocode_cluster_metrics.py` — silhouette / Calinski-Harabasz /
  Davies-Bouldin / inertia + normalization + `kneed` elbow. All are closed-form over the
  distance matrix + labels; port formulas + Kneedle. moderate
- `src/dataframes/geocode_silhouette_score.py` — per-sample silhouette. ✅ (same math)
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