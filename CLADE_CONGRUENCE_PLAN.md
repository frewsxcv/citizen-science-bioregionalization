# Clade Congruence Plan

Productionizing "do birds and beetles draw the same map?" — a web interface for
comparing data-derived bioregionalizations across taxonomic scopes (kingdom,
phylum, class, order) at global extent.

## TL;DR

- The facet dimension **does not exist yet**. `build_taxon_filter()` raises
  `NotImplementedError` (`src/darwin_core_utils.py:221`) and
  `build_darwin_core_lf()` accepts `taxon_filter`, logs it, and never applies it
  (`src/dataframes/darwin_core.py:24`). This is greenfield, not a tweak.
- Phase 1 is a **kill gate**, not a deliverable. Shared spatial sampling bias
  across clades can manufacture apparent congruence. If observed agreement does
  not exceed a null model, stop — everything downstream is wasted.
- The affordability move is a **facet-agnostic snapshot intermediate**: aggregate
  the GBIF snapshot once per release, then serve every facet from it. Turns N
  full scans into 1.
- The scaling move is **client-side comparison**: ship compact per-facet cluster
  assignment vectors over a canonical hex ordering and diff them in the browser.
  150 facets is 150 artifacts, not 11,175 precomputed pairs.
- Global clustering is blocked on the dense `pdist` distance matrix
  (`src/matrices/geocode_distance.py:208`). H3 r4 globally is ~288k cells → ~41B
  condensed pairs. Does not fit, at any machine size.

---

## Kill criteria (read before building anything)

The scientific risk dominates the engineering risk. Both Aves and Coleoptera are
oversampled in Western Europe and North America and undersampled across Central
Africa and Siberia. Both will therefore exhibit structure driven by observer
density, and **they will agree on it**. A naive pipeline measures "where people
look" and reports it as "where biogeographic barriers are."

Phase 1 exists solely to answer: *does clade-pair agreement exceed an
effort-matched null?*

- **Yes, clearly** → proceed to Phase 2.
- **Marginal** → the product may still work but only with rarefaction and
  null-relative reporting baked in from the start; revisit scope.
- **No** → stop. Do not build Phases 2-5.

---

## Current state

### What exists and works

| Capability | Location |
| --- | --- |
| H3 geocoding (UInt64 cells, `polars_h3`) | `src/geocode.py` |
| Connectivity-constrained Ward clustering, all k from one merge tree | `bioregion_rs.build_geocode_cluster_multi_k` via `src/dataframes/geocode_cluster.py:101` |
| Neighbor graph w/ single-connected-component guarantee | `bioregion_rs.build_geocode_neighbors` via `src/dataframes/geocode_neighbors.py` |
| Elbow-method k selection (Kneedle) | `src/cluster_optimization.py` |
| Per-hex silhouette scores | `src/dataframes/geocode_silhouette_score.py` |
| PERMANOVA validity test | `src/dataframes/permanova_results.py` |
| Indicator taxa, log2 fold change vs. geographic neighbors | `src/dataframes/cluster_significant_differences.py` |
| Composition-space cluster coloring (`color_method="taxonomic"`) | `src/dataframes/cluster_color.py` |
| MapLibre + React + zustand frontend, static Parcel build | `frontend/` |

### What does not exist

- **Taxonomic filtering.** See TL;DR. Rank keys (`kingdomKey`, `classKey`,
  `orderKey`, …) appear in `_LOWER_TO_CAMEL` (`src/darwin_core_utils.py:52-56`)
  but not in `_BASE_SCHEMA`, so they are never loaded or selected.
- **Any batch/headless path.** Deployment is a single GCE box running
  `marimo edit` on :8080 (`terraform/main.tf`, `Dockerfile` CMD).
- **Any artifact store or run manifest.** Output is `output/output.geojson`,
  overwritten per run (`src/output.py`).
- **Cross-facet anything** — shared hex ordering, stable colors, comparison
  metrics.

### Known correctness issues to fix en route

1. **~~Bray-Curtis on scaled data.~~** ✅ ADDRESSED. `pdist(metric="braycurtis")`
   ran on a `RobustScaler`-transformed matrix. RobustScaler centers on the
   median and emits negatives; Bray-Curtis is defined for non-negative
   abundances. The distortion is *fit per-facet*, so it corrupted exactly the
   cross-facet comparability this feature depends on.

   `GeocodeDistanceMatrix.build` now takes a `metric` argument
   (`src/types.py:CompositionMetric`). `"abundance"` is the unchanged
   historical path; `"presence"` binarizes to presence/absence and uses
   Sorensen (scipy's `dice`), which is fit-free — a geocode's vector depends
   only on which taxa it contains, never on the rest of the dataset. Euclidean
   is used on the UMAP embedding for the presence path, which is also what Ward
   linkage assumes.

   Two latent defects surfaced while building this, both fixed:
   - **UMAP severed maximal-distance edges.** UMAP defaults
     `disconnection_distance` to 1.0 for bounded metrics including `dice`, and
     Sorensen distance is *exactly* 1 whenever two geocodes share no taxa —
     routine in sparse occurrence data. Left at the default, UMAP dropped those
     edges, fully disconnected the affected geocodes, and emitted NaN
     coordinates that propagated through `pdist` and Ward into the cluster
     metrics **without raising**. Observed on the sample archive: 40 of 55
     pairwise distances NaN, every silhouette score NaN. Now pinned to
     infinity (a no-op for braycurtis, whose default is already infinite).
   - **Unstable feature-matrix column order.** `pivot_taxon_counts` took its
     column list from `.unique()`, which promises no ordering, so columns
     shuffled between runs. Distances are permutation-invariant so results
     stayed ~stable, but it made seeded runs irreproducible. Now sorted.

   `random_state` is now plumbed through to UMAP. **Phase 1 step 4 needs
   this**: without a pinned seed, run-to-run UMAP noise is indistinguishable
   from the null-model variation being measured.

   Harness: `scripts/compare_composition_metrics.py` builds everything upstream
   of the distance matrix once, then clusters both metrics over it and reports
   per-k ARI/AMI plus each metric's silhouette. **Still needs a real run** — it
   has only been smoke-tested against the 11-hex sample archive, which is far
   too small and too skewed to validate anything.
2. **Dense distance matrix.** `pdist`/`squareform` is O(n²). Blocking for global
   extent (Phase 5), not for the Phase 1 continental spike.
3. **~~`Dockerfile` COPY of a missing file~~ — the image could never build at
   all.** ✅ FIXED. Investigation found four independent breaks, not one:
   - `COPY ocean.geojson .` referenced a file deleted in `c3f6bc2`
     (2026-01-11, *"remove ocean.geojson and references"*), which cleaned up
     `notebook.py`, `cluster_color.py`, `geojson.py` and tests but missed the
     Dockerfile.
   - `uv pip install -e .` cannot work on this project: the root is a **virtual**
     project (`source = { virtual = "." }` in `uv.lock`) and is not installable.
   - `bioregion-rs` is an editable uv workspace member, but `bioregion_rs/` was
     never copied into the build context.
   - `python:3.13-slim` has no Rust toolchain, so the maturin extension could not
     have compiled even with the sources present.

   Additionally `.dockerignore` excluded `README.md` (referenced by
   `[project].readme`, so resolution fails without it) and `uv.lock` (needed by
   `uv sync --frozen`), under a stale "using requirements.txt instead" comment —
   there is no `requirements.txt`.

   Replaced with a two-stage build (Rust builder → slim runtime carrying only the
   built virtualenv). **No CI workflow built the image**, which is why this stayed
   invisible for ~6 months with all four workflows green; added
   `.github/workflows/docker.yml` with a build plus a smoke test that loads the
   compiled extension and resolves a scope from the packaged registry.

---

## Target architecture

```
GBIF snapshot (parquet, ~3B rows)
        │
        │  ONE job per snapshot release
        ▼
snapshot intermediate  (h3_r5, taxonKey, rank keys) → count
        │  partitioned by kingdomKey; ~10^8 rows, few GB
        │
        │  cheap filtered read + H3 parent rollup per facet
        ▼
facet run (scope × h3_res × extent)  ──► artifact:
        manifest.json      provenance, PERMANOVA p, support size
        clusters.parquet   hex → cluster, ALL k in range
        assignments.bin    dense Uint16 over canonical hex ordering
        taxa.parquet       indicator taxa per cluster
        silhouette.parquet
        │
        ▼
shared static assets (fetched once by client)
        hex_index.bin      canonical global H3 ordering
        edges.bin          neighbor pairs as index pairs
        │
        ▼
browser: fetch 2 assignment vectors → diff over edges → render
```

The client-side diff is the load-bearing decision. Precomputing tiles for all
pairs of 150 orders is 11,175 combinations; precomputing per-facet vectors is
150. Global r4 as `Uint16Array` is ~576KB raw and gzips hard (cluster labels are
spatially coherent).

---

## Phase 0 — Facet dimension — ✅ DONE

**Goal:** `taxon_filter` actually filters. No comparison logic yet.

**Delivered:** `TaxonScope` (`src/types.py`), offline key registry
(`src/data/taxon_keys.json` + `scripts/fetch_taxon_keys.py`), scope parsing
(`src/taxon_scope.py`), key-based filtering wired through
`build_darwin_core_lf`, and `--scope=rank:name` on the CLI/notebook. 24 new tests
in `test/test_taxon_scope.py`; full suite (98) green, pyright clean.

**Verified:** over the sample archive at r9, `kingdom:Animalia` yields 240 hexes
and `kingdom:Plantae` 45, sharing only 34 — different data and different cluster
boundaries, as required.

**Findings worth carrying forward:**

- **Scope must be applied before `limit`.** Limiting first and scoping second
  yields near-zero rows for any selective scope. Now enforced and regression
  tested; the batch runner must preserve this ordering.
- **GBIF backbone rank assignments differ from textbook taxonomy.** `Squamata`
  and `Testudines` are backbone *classes*; `Reptilia` (paraphyletic) and
  `Actinopterygii` are absent entirely, and `Salmo salar` currently resolves with
  a null class. The Phase 4 facet picker must therefore be driven by what the
  backbone and data actually contain, not by a hand-written taxonomy — and the
  curated list needs re-validation after each backbone release.
- **Sparse scopes break UMAP before they break clustering.**
  `umap_n_components = height - 2` (`src/matrices/geocode_distance.py:191`), so a
  scope resolving to ≤2 hexes fails with a bare `n_components must be greater
  than 0`. Selective scopes hit this routinely. Phase 2's validity floor should
  reject such facets up front with a clear message instead.

1. Add rank keys to `_BASE_SCHEMA` (`src/darwin_core_utils.py`): `kingdomKey`,
   `phylumKey`, `classKey`, `orderKey`, `familyKey` as `pl.UInt32`.
2. Replace `build_taxon_filter(taxon_name: str)` with a typed scope:
   ```python
   @dataclass(frozen=True)
   class TaxonScope:
       rank: Literal["kingdom", "phylum", "class", "order", "family"]
       key: int          # GBIF backbone key, NOT a name string
       label: str        # display only, e.g. "Aves"
   ```
   Filter on the integer key. Rationale: stable across GBIF backbone renames,
   and the predicate pushes into the parquet scan for row-group pruning — a
   string comparison against a name column will not prune comparably.
3. Wire it through `build_darwin_core_lf()` — apply the filter, and add the
   scope's rank column to the `.select()`.
4. Add a name→key resolver (GBIF species API, cached to a checked-in JSON) so the
   CLI and UI can accept "Aves".
5. Update `--taxon-filter` CLI parsing and the marimo UI element to
   `--scope=order:Coleoptera` form.

**Exit:** two runs over the same bbox with different scopes produce demonstrably
different geocode counts and cluster boundaries.

---

## Phase 1 — Validation spike (KILL GATE)

**Goal:** answer the scientific question. Deliberately not productionized —
notebook-grade code is fine, throw it away afterward.

**Scope:** Aves vs. Coleoptera, H3 r3, one continent (reuse the existing Eastern
US default bbox or widen to North America). Small enough that the dense distance
matrix is not yet a problem.

1. ~~Fix the Bray-Curtis/RobustScaler conflict.~~ 🟡 CODE LANDED, VALIDATION
   PENDING. Cross-facet work now runs **presence/absence with Sørensen** via
   `metric="presence"`; see "Known correctness issues" above for what changed
   and for two latent defects found en route. Substantially less
   effort-sensitive; discards abundance information, which is an acceptable
   trade for comparability.

   **The validation itself has not been done.**
   `scripts/compare_composition_metrics.py` is written and smoke-tested, but
   only against the 11-hex sample archive. Answering "does Sørensen reproduce
   the existing single-facet result?" needs a real GBIF run at r3 over a
   continent. Do that before step 2 — if the two metrics disagree badly, every
   later cross-facet number is reporting the metric switch rather than biology.
2. Build the **comparison frame**:
   - **Common support** — cluster both facets only on hexes clearing a minimum
     observation threshold *in both*. Comparing over the union measures data
     availability, not biology.
   - **Fixed k** — pin k across both facets. ARI between a 6-cluster and a
     19-cluster partition is low for combinatorial reasons alone. The merge tree
     already yields all k free; persist the range.
   - **Rarefaction** — resample each hex to a fixed observation count, R
     repetitions, consensus clustering.
3. Compute:
   - **Scalar:** Adjusted Rand Index and Adjusted Mutual Information over common
     support.
   - **Edge layer:** for each adjacent hex pair, classify as shared-barrier /
     A-only / B-only / shared-interior.
4. Compute the **null**: rebuild facet B by sampling taxa from the whole pool
   matching B's per-hex effort profile, recluster, recompute ARI. ~100 draws.
   The reportable quantity is observed-minus-null, never raw ARI.
5. Sanity check against known biogeography — a North America run should recover
   the eastern/western forest split and the Great Plains transition. If it
   recovers nothing recognizable, that is also a kill signal.

**Exit:** a written go/no-go with the null-relative effect size. Estimated ~1
week.

---

## Phase 2 — Snapshot intermediate + batch runner

**Goal:** make facet runs cheap and repeatable. Only after Phase 1 says go.

1. **Snapshot aggregation job** (new module, e.g. `src/snapshot/aggregate.py`):
   one pass over the GBIF parquet producing
   `(h3_r5, taxonKey, kingdomKey, phylumKey, classKey, orderKey, familyKey) → count`,
   partitioned by `kingdomKey`. Aggregate at the finest resolution ever served;
   r5→r4→r3 rollup is integer arithmetic on H3 cell IDs, no re-geocoding.
2. **Headless CLI entrypoint** that calls `src/` directly, bypassing marimo. The
   `marimo run notebook.py --` form works for one-offs but is the wrong layer for
   hundreds of runs.
3. **Content-addressed artifacts.** Key each run by hash of
   `(snapshot_id, scope, h3_res, extent, k_range, effort_params)`. Immutable,
   cache-reusable, and re-runs only what changed.
4. **Run manifest** carrying provenance: GBIF snapshot date, PERMANOVA p-value,
   common-support size, silhouette distribution, rarefaction params.
5. **Batch execution** — Cloud Run Jobs or GCP Batch, parallel over the facet
   list, reading the intermediate. Reuse the Docker image with the marimo CMD
   stripped (and the `ocean.geojson` COPY removed).

**Exit:** ~10 curated clades at r3, global-ish extent, reproducible from a single
command.

---

## Phase 3 — Artifact model & congruence computation

1. **Canonical hex ordering.** One sorted global H3 cell list per resolution,
   published as a shared static asset. Every facet's assignment vector indexes
   into it. This is the contract that makes client-side diffing possible.
2. **Shared neighbor edge list** as index pairs into that ordering, published
   alongside. `geocode_cluster.py:101` already derives the upper-triangle edge
   list for clustering — reuse the same derivation.
3. **Dense assignment export** — `Uint16` per hex per k, with a sentinel for
   "below support threshold."
4. **Validity floor.** The UI must refuse to render facets failing a PERMANOVA /
   silhouette threshold rather than displaying statistically meaningless
   clusters. Encode the floor in the manifest.
5. **Pairwise scalars** for all facet pairs — cheap label-vector comparisons,
   seconds each. Store as a matrix artifact.

---

## Phase 4 — Frontend compare mode

1. **Facet picker** — taxonomic tree with occurrence counts; scopes below the
   data floor rendered disabled with the reason shown. Being explicit about
   "Isopoda has insufficient data to regionalize globally" builds more trust than
   rendering noise.
2. **Dual synced maps** (or swipe) with a shared viewport.
3. **Client-side edge diff** — fetch two assignment vectors, loop the shared edge
   list, emit the four-way boundary classification as a rendered layer. Thick
   strokes for shared barriers.
4. **Shared k slider** driving both facets simultaneously.
5. **Cross-facet stable color.** Cluster IDs are arbitrary per run, so naive
   coloring makes facet-A-region-3 and facet-B-region-3 unrelated but identically
   colored — which destroys the entire comparison. Extend
   `cluster_color.py`'s `taxonomic` method to a *shared* embedding fitted once
   across facets rather than per-run.
6. **URL as state** — `/?a=class:Aves&b=order:Coleoptera&res=3&k=8#3.2/-40/145`.
   Every comparison is a citable link. Non-negotiable for research and sharing.
7. **Null-relative framing in the UI.** Display the effect size over null, not
   raw agreement, with the common-support fraction shown alongside.

---

## Phase 5 — Scale-out

1. **Sparse distance computation.** Blocking for true global extent. Two viable
   routes:
   - Restrict distances to the connectivity graph already built by
     `build_geocode_neighbors`, feeding a sparse-aware Ward implementation.
   - Hierarchical divide-and-conquer: cluster within continents, then merge.

   The current Eastern-US default bbox (`src/defaults.py`) is quietly hiding this
   ceiling.
2. **Every order clearing the data floor** (~150 realistic globally).
3. **Congruence matrix view** — cluster the clade × clade agreement matrix to
   surface which clades share biogeographic structure. This is a genuinely novel
   artifact and the strongest landing-page visual.
4. **Monthly refresh** on GBIF snapshot release: one aggregation job → facet
   fan-out → scalar matrix → publish.

---

## Risks / open questions

- **Sampling bias dominates signal.** The central risk; Phase 1 exists to
  measure it. Mitigations (rarefaction, presence/absence, null-relative
  reporting) reduce but never eliminate it.
- **Rarefaction cost.** R repetitions × consensus clustering multiplies facet
  runtime by R. May force a coarser default resolution or a smaller R than is
  statistically comfortable.
- **Presence/absence vs. abundance.** Recommended for comparability, but discards
  real signal and may weaken boundaries in genuinely abundance-structured
  systems. Worth an explicit A/B in Phase 1.
- **k selection interacts with facet richness.** Fixed k is right for pairwise
  comparison but arguably wrong for "what are this clade's true regions."
  Possibly ship both modes.
- **Marine vs. terrestrial.** H3 cells over ocean behave differently for marine
  clades; the `no_edges` machinery
  (`build_geocode_neighbors_no_edges_df`) handles boundary hexes but not the
  land/sea distinction. Undecided.
- **GBIF backbone drift** across snapshots changes rank keys and taxon
  identities, breaking temporal comparison. Needs a key-stability audit before
  any time-series feature.
- **Facet pair count.** Client-side diffing solves precompute, but the all-pairs
  scalar matrix is still O(n²) = ~11k for 150 orders. Cheap per pair, but the
  null-model runs are not — likely restrict nulls to curated pairs.
