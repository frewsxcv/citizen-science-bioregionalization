"""Micro-benchmark: Rust (bioregion_rs) vs the original Python/library
implementations for the compute-heavy ported pipeline stages.

Rationale: the sample archive (~165 geocodes) is too small to show anything --
every stage finishes in single-digit milliseconds and total runtime is
dominated by imports/UMAP/HTML export. The migration's value is in the
O(n^2)/O(n^3) stages at gbif-snapshot scale (thousands of geocodes), which
can't be run locally. So we synthesize inputs at controlled scales and time
each ported hotspot against exactly the library call the original code used.

Companion to harness.py: harness.py checks Rust vs Python for *correctness*,
this checks them for *speed*. The Python baselines here are exactly the library
calls the pre-migration code used (sklearn / skbio), so the numbers reflect what
the migration actually bought each stage.

Run: uv run python bioregion_rs/benchmark.py
"""

import time
from statistics import median

import numpy as np
import polars as pl
from scipy.spatial.distance import pdist, squareform

import bioregion_rs

RNG = np.random.default_rng(1234)


def bench(fn, repeats):
    """Return median wall-clock seconds over `repeats` runs (1 warmup discarded)."""
    fn()  # warmup
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return median(times)


def grid_graph(n):
    """n points on a jittered 2D grid with 4-neighbour adjacency (connected),
    approximating the local hex-adjacency of real geocodes. Returns
    (coords, condensed_distances, edges[list of (a,b) with a<b])."""
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    pts = []
    idx_of = {}
    for r in range(rows):
        for c in range(cols):
            if len(pts) >= n:
                break
            idx_of[(r, c)] = len(pts)
            pts.append((r + RNG.uniform(-0.2, 0.2), c + RNG.uniform(-0.2, 0.2)))
    coords = np.array(pts)
    edges = []
    for (r, c), i in idx_of.items():
        for dr, dc in ((1, 0), (0, 1)):
            j = idx_of.get((r + dr, c + dc))
            if j is not None:
                edges.append((i, j))
    condensed = pdist(coords)
    return coords, condensed, edges


# --------------------------------------------------------------------------
# Stage 1: connectivity-constrained Ward clustering, per k in [2, max_k]
# Rust: bioregion_rs.build_geocode_cluster_multi_k (full tree once, cut per k)
# Python: sklearn AgglomerativeClustering(ward, connectivity=...).fit_predict,
#         re-fit once per k -- exactly the original geocode_cluster.py loop.
# --------------------------------------------------------------------------
def bench_ward(n, max_k, repeats):
    from scipy.sparse import csr_matrix
    from sklearn.cluster import AgglomerativeClustering

    coords, condensed, edges = grid_graph(n)
    geocodes = list(range(n))
    ea = [a for a, b in edges]
    eb = [b for a, b in edges]
    condlist = condensed.tolist()

    square = squareform(condensed)
    conn = np.zeros((n, n), dtype=np.int64)
    for a, b in edges:
        conn[a, b] = conn[b, a] = 1
    conn_csr = csr_matrix(conn)

    def rust():
        bioregion_rs.build_geocode_cluster_multi_k(geocodes, condlist, ea, eb, 2, max_k)

    def py():
        for k in range(2, max_k + 1):
            AgglomerativeClustering(
                n_clusters=k, connectivity=conn_csr, linkage="ward"
            ).fit_predict(square)

    return bench(rust, repeats), bench(py, max(1, repeats // 2))


# --------------------------------------------------------------------------
# Stage 2: PERMANOVA (pseudo-F + permutation p-value)
# Rust: bioregion_rs.build_permanova_results
# Python: skbio.stats.distance.permanova -- the original library call.
# --------------------------------------------------------------------------
def bench_permanova(n, n_clusters, permutations, repeats):
    from skbio.stats.distance import DistanceMatrix, permanova

    _, condensed, _ = grid_graph(n)
    condlist = condensed.tolist()
    geocode_ids = list(range(n))
    labels = (RNG.integers(0, n_clusters, size=n)).tolist()
    cluster_df = pl.DataFrame(
        {"geocode": geocode_ids, "cluster": labels}
    ).cast({"geocode": pl.UInt64, "cluster": pl.UInt32})

    sq = squareform(condensed)
    ids = [str(i) for i in geocode_ids]
    dm = DistanceMatrix(sq, ids=ids)

    def rust():
        bioregion_rs.build_permanova_results(condlist, geocode_ids, cluster_df, permutations)

    def py():
        permanova(dm, labels, permutations=permutations)

    return bench(rust, repeats), bench(py, repeats)


# --------------------------------------------------------------------------
# Stage 3: cluster metrics (silhouette / Calinski-Harabasz / Davies-Bouldin /
# inertia) for every k. Rust: bioregion_rs.build_geocode_cluster_metrics (one
# call, all k). Python: the original per-k loop over sklearn.metrics + a custom
# inertia. Both consume real Rust ward output (matching the pipeline dataflow).
# --------------------------------------------------------------------------
def _compute_inertia(dm_square, labels):
    total = 0.0
    for label in np.unique(labels):
        idx = np.where(labels == label)[0]
        if len(idx) <= 1:
            continue
        d = dm_square[np.ix_(idx, idx)]
        total += np.sum(d**2) / (2 * len(idx))
    return float(total)


def bench_metrics(n, max_k, repeats):
    from sklearn.metrics import (
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
    )

    _, condensed, edges = grid_graph(n)
    condlist = condensed.tolist()
    geocodes = list(range(n))
    ea = [a for a, b in edges]
    eb = [b for a, b in edges]
    # Real multi-k clustering to feed both metric implementations.
    multi_k = bioregion_rs.build_geocode_cluster_multi_k(geocodes, condlist, ea, eb, 2, max_k)
    square = squareform(condensed)

    per_k = {}
    for k in range(2, max_k + 1):
        per_k[k] = (
            multi_k.filter(pl.col("num_clusters") == k).sort("geocode")["cluster"].to_numpy()
        )

    def rust():
        bioregion_rs.build_geocode_cluster_metrics(condlist, multi_k)

    def py():
        for k in range(2, max_k + 1):
            labels = per_k[k]
            silhouette_score(square, labels=labels, metric="precomputed")
            calinski_harabasz_score(square, labels)
            davies_bouldin_score(square, labels)
            _compute_inertia(square, labels)

    return bench(rust, repeats), bench(py, repeats)


def fmt(rs, py):
    speedup = py / rs if rs > 0 else float("inf")
    return f"{rs*1000:9.2f} ms   {py*1000:10.2f} ms   {speedup:7.1f}x"


def main():
    print(f"polars {pl.__version__}\n")

    print("=" * 74)
    print("Ward clustering  (multi-k, k=2..max_k; connectivity-constrained)")
    print("  Rust: build tree once, cut per k   |   Python: sklearn re-fit per k")
    print("=" * 74)
    print(f"{'n_geocodes':>10} {'max_k':>6}   {'Rust':>10}   {'Python':>12}   {'speedup':>8}")
    for n, max_k, reps in [(150, 10, 5), (400, 15, 3), (800, 15, 2), (1500, 15, 1)]:
        rs, py = bench_ward(n, max_k, reps)
        print(f"{n:>10} {max_k:>6}   {fmt(rs, py)}")

    print()
    print("=" * 74)
    print("PERMANOVA  (pseudo-F + permutation test)")
    print("  Rust: bioregion_rs   |   Python: skbio.stats.distance.permanova")
    print("=" * 74)
    print(f"{'n_geocodes':>10} {'perms':>6}   {'Rust':>10}   {'Python':>12}   {'speedup':>8}")
    for n, perms, reps in [(150, 999, 5), (400, 999, 3), (800, 999, 2), (1500, 999, 1)]:
        rs, py = bench_permanova(n, 8, perms, reps)
        print(f"{n:>10} {perms:>6}   {fmt(rs, py)}")

    print()
    print("=" * 74)
    print("Cluster metrics  (silhouette / CH / DB / inertia, per k)")
    print("  Rust: one call, all k   |   Python: per-k loop over sklearn.metrics")
    print("=" * 74)
    print(f"{'n_geocodes':>10} {'max_k':>6}   {'Rust':>10}   {'Python':>12}   {'speedup':>8}")
    for n, max_k, reps in [(150, 10, 5), (400, 15, 3), (800, 15, 2), (1500, 15, 1)]:
        rs, py = bench_metrics(n, max_k, reps)
        print(f"{n:>10} {max_k:>6}   {fmt(rs, py)}")


if __name__ == "__main__":
    main()
