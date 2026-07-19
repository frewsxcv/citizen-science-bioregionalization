"""Correctness harness for the Rust ports.

Every ported function is now the live pipeline implementation (the Python
call sites delegate to `bioregion_rs`), so a Rust-vs-our-own-Python diff would
just compare Rust against itself. This harness keeps only the checks that
validate Rust against something *independent*:

  - an external reference library: skbio (PERMANOVA), sklearn
    (connectivity-constrained Ward clustering), polars_h3 (geocoding);
  - a hand-written Python reference: darken_hex_color;
  - the two stages still implemented in Python in the live pipeline, whose
    Rust ports exist but are not cut over: taxonomy and geocode_taxa_counts.

(Fisher's exact, the YlOrRd palette, kneed, robust scaling, etc. are validated
against their references directly in the crate's `cargo test` unit tests.)

Run:  uv run python bioregion_rs/harness.py
"""

import math
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# Make the repo-root `src` package importable regardless of CWD.
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import polars as pl

import bioregion_rs
from src.colors import darken_hex_color as py_darken
from src.dataframes.darwin_core import build_darwin_core_lf
from src.dataframes.geocode import build_geocode_df
from src.dataframes.geocode_taxa_counts import build_geocode_taxa_counts_lf
from src.dataframes.permanova_results import build_permanova_results_df
from src.dataframes.taxonomy import build_taxonomy_lf
from src.geocode import (
    filter_by_bounding_box as py_filter_bbox,
    select_geocode_lf,
    with_geocode_lf,
)
from src.matrices.geocode_connectivity import GeocodeConnectivityMatrix
from src.matrices.geocode_distance import GeocodeDistanceMatrix
from src.types import Bbox


def check(name: str, ok: bool) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        raise SystemExit(f"harness failed at: {name}")


def sample_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "decimalLatitude": [37.77, -33.87, 0.0, 51.5, None, 90.0],
            "decimalLongitude": [-122.42, 151.21, 0.0, -0.12, 10.0, None],
        }
    )


# --- src/colors.py -----------------------------------------------------------


def test_darken_hex_color() -> None:
    print("src/colors.py  (darken_hex_color):")
    cases = ["#ff0000", "#00ff00", "#123456", "#f0a", "#ffffff", "#000000"]
    for factor in (0.5, 0.3, 0.75, 1.0):
        for c in cases:
            check(
                f"{c} @ {factor}",
                bioregion_rs.darken_hex_color(c, factor) == py_darken(c, factor),
            )


# --- src/geocode.py ----------------------------------------------------------


def test_select_geocode() -> None:
    print("src/geocode.py  (select_geocode):")
    df = sample_frame()
    for precision in (2, 4, 6, 8):
        rust_out = bioregion_rs.select_geocode(df, precision)
        py_out = select_geocode_lf(df.lazy(), geocode_precision=precision).collect()
        check(
            f"precision={precision} is pl.DataFrame", isinstance(rust_out, pl.DataFrame)
        )
        check(
            f"precision={precision} geocode matches polars_h3",
            rust_out["geocode"].equals(py_out["geocode"]),
        )


def test_with_geocode() -> None:
    print("src/geocode.py  (with_geocode):")
    df = sample_frame()
    for precision in (4, 8):
        rust_out = bioregion_rs.with_geocode(df, precision)
        py_out = with_geocode_lf(df.lazy(), precision).collect()
        check(f"precision={precision} full frame matches", rust_out.equals(py_out))


def test_filter_by_bounding_box() -> None:
    print("src/geocode.py  (filter_by_bounding_box):")
    df = sample_frame()
    bbox = Bbox.from_coordinates(0.0, 60.0, -130.0, 20.0)
    rust_out = bioregion_rs.filter_by_bounding_box(
        df, bbox.min_lat, bbox.max_lat, bbox.min_lng, bbox.max_lng
    )
    py_out = py_filter_bbox(df.lazy(), bounding_box=bbox).collect()
    check("filtered frame matches", rust_out.equals(py_out))


def _load_bbox_darwin_and_geocode_no_edges():
    """Shared fixture: sample-archive Darwin Core data + its non-edge geocodes."""
    load_bbox = Bbox.from_coordinates(-90.0, 90.0, -180.0, 180.0)
    bbox = Bbox.from_coordinates(35.0, 43.5, -50.0, 10.0)
    precision = 4
    darwin_lf = build_darwin_core_lf(
        str(REPO_ROOT / "test" / "sample-archive"), load_bbox
    )
    darwin_df = darwin_lf.collect()
    geocode_df = build_geocode_df(darwin_lf, precision, bbox)
    geocode_no_edges_df = geocode_df.filter(~pl.col("is_edge"))
    return bbox, precision, darwin_lf, darwin_df, geocode_no_edges_df


def _taxonomy_pairs(taxonomy_df: pl.DataFrame) -> set:
    return set(taxonomy_df.select("scientificName", "gbifTaxonId").iter_rows())


def test_build_taxonomy() -> None:
    print("src/dataframes/taxonomy.py  (build_taxonomy):")
    bbox, precision, _darwin_lf, darwin_df, geocode_no_edges_df = (
        _load_bbox_darwin_and_geocode_no_edges()
    )

    rust_out = bioregion_rs.build_taxonomy(
        darwin_df,
        precision,
        geocode_no_edges_df,
        bbox.min_lat,
        bbox.max_lat,
        bbox.min_lng,
        bbox.max_lng,
    )
    py_out = build_taxonomy_lf(
        darwin_df.lazy(), precision, geocode_no_edges_df.lazy(), bbox
    ).collect()

    check(f"non-trivial: {py_out.height} taxa built", py_out.height > 0)
    check(
        "same (scientificName, gbifTaxonId) pair set",
        _taxonomy_pairs(rust_out) == _taxonomy_pairs(py_out),
    )
    check(
        "taxonId is a 0..n bijection (rust)",
        set(rust_out["taxonId"].to_list()) == set(range(rust_out.height)),
    )
    check(
        "taxonId is a 0..n bijection (python)",
        set(py_out["taxonId"].to_list()) == set(range(py_out.height)),
    )


def test_build_geocode_taxa_counts() -> None:
    print("src/dataframes/geocode_taxa_counts.py  (build_geocode_taxa_counts):")
    bbox, precision, _darwin_lf, darwin_df, geocode_no_edges_df = (
        _load_bbox_darwin_and_geocode_no_edges()
    )

    rust_taxonomy = bioregion_rs.build_taxonomy(
        darwin_df,
        precision,
        geocode_no_edges_df,
        bbox.min_lat,
        bbox.max_lat,
        bbox.min_lng,
        bbox.max_lng,
    )
    py_taxonomy = build_taxonomy_lf(
        darwin_df.lazy(), precision, geocode_no_edges_df.lazy(), bbox
    ).collect()

    rust_counts = bioregion_rs.build_geocode_taxa_counts(
        darwin_df,
        precision,
        rust_taxonomy,
        geocode_no_edges_df,
        bbox.min_lat,
        bbox.max_lat,
        bbox.min_lng,
        bbox.max_lng,
    )
    py_counts = build_geocode_taxa_counts_lf(
        darwin_df.lazy(),
        precision,
        py_taxonomy.lazy(),
        geocode_no_edges_df.lazy(),
        bbox,
    ).collect()

    check(
        f"non-trivial: {py_counts.height} (geocode, taxon) rows built",
        py_counts.height > 0,
    )
    check(
        "total count matches",
        int(rust_counts["count"].sum()) == int(py_counts["count"].sum()),
    )

    # taxonId assignment order can differ between engines (unique() ordering is
    # not guaranteed), so compare by joining back through each engine's own
    # taxonomy to (geocode, scientificName, gbifTaxonId, count).
    def _resolved(counts: pl.DataFrame, taxonomy: pl.DataFrame) -> set:
        joined = counts.join(taxonomy, on="taxonId", how="left")
        return set(
            joined.select(
                "geocode", "scientificName", "gbifTaxonId", "count"
            ).iter_rows()
        )

    check(
        "same (geocode, scientificName, gbifTaxonId, count) set",
        _resolved(rust_counts, rust_taxonomy) == _resolved(py_counts, py_taxonomy),
    )


def test_build_permanova_results() -> None:
    print("src/dataframes/permanova_results.py  (build_permanova_results):")
    # Hand-built (not sample-archive-derived, which is too small for a
    # meaningful permutation test): 3 geocodes tightly clustered in group 0,
    # 3 in group 1, far apart from group 0.
    geocode_ids = [1, 2, 3, 4, 5, 6]
    condensed = [
        0.1, 0.1, 0.9, 0.9, 0.9,  # (1,2) (1,3) (1,4) (1,5) (1,6)
             0.1, 0.9, 0.9, 0.9,  # (2,3) (2,4) (2,5) (2,6)
                  0.9, 0.9, 0.9,  # (3,4) (3,5) (3,6)
                       0.1, 0.1,  # (4,5) (4,6)
                            0.1,  # (5,6)
    ]  # fmt: skip
    geocode_cluster_df = pl.DataFrame(
        {"geocode": geocode_ids, "cluster": [0, 0, 0, 1, 1, 1]}
    ).cast({"geocode": pl.UInt64, "cluster": pl.UInt32})
    distance_matrix = GeocodeDistanceMatrix(
        condensed=np.array(condensed), reduced_features=np.empty((6, 0))
    )
    geocode_lf = (
        pl.DataFrame({"geocode": geocode_ids}).cast({"geocode": pl.UInt64}).lazy()
    )

    # permutations=0 skips the Monte Carlo step entirely, so this is fully
    # deterministic -- but dataframely's schema rejects the resulting NaN
    # p_value (build_permanova_results_df's own output would fail its own
    # validation), so call skbio directly for this exact-match check rather
    # than through the wrapper function.
    from skbio.stats.distance import DistanceMatrix, permanova as skbio_permanova

    dm_skbio = DistanceMatrix(np.array(condensed), ids=[str(g) for g in geocode_ids])
    skbio_res = skbio_permanova(dm_skbio, [0, 0, 0, 1, 1, 1], permutations=0)
    py_test_statistic = skbio_res["test statistic"]

    rust_out = bioregion_rs.build_permanova_results(
        condensed, geocode_ids, geocode_cluster_df, 0
    )
    check(
        "test_statistic matches skbio exactly",
        abs(rust_out["test_statistic"][0] - py_test_statistic) < 1e-9,
    )
    check("p_value is NaN (permutations=0)", math.isnan(rust_out["p_value"][0]))

    # permutations=999: skbio.stats.distance.permanova runs its Monte Carlo
    # permutation test with an unseeded RNG (seed=None), so the p-value isn't
    # bit-reproducible even between two separate Python runs, let alone
    # between Python and Rust -- only statistical equivalence is achievable.
    # With only 6 objects in 2 equal-size groups there are just 20 distinct
    # 3-3 partitions, several tied with (or close to) the observed one given
    # this fixture's symmetric 0.1/0.9 distances, so the true p-value sits
    # around ~0.10 (not near 0 as "obviously separated" clusters might
    # suggest). Check both estimates land within a few standard errors of
    # each other (SE ~= sqrt(0.1*0.9/999) ~= 0.0095 here) instead of asserting
    # a specific significance threshold.
    rust_out2 = bioregion_rs.build_permanova_results(
        condensed, geocode_ids, geocode_cluster_df, 999
    )
    py_out2 = build_permanova_results_df(
        distance_matrix, geocode_cluster_df, geocode_lf, permutations=999
    )
    rust_p = rust_out2["p_value"][0]
    py_p = py_out2["p_value"][0]
    check(
        f"p-values statistically consistent (rust={rust_p:.4f}, py={py_p:.4f})",
        abs(rust_p - py_p) < 0.05,
    )


def _grid_cluster_fixture():
    """A connected 4x5 spatial grid graph over 20 points with *integer*
    coordinates. Integer coords make many pairwise distances (hence many ward
    inertias) exactly equal, which stresses the merge-heap tie-breaking that
    both sklearn and the Rust port resolve via the (inertia, row, col) key."""
    from scipy.spatial.distance import pdist

    from src.matrices.geocode_connectivity import GeocodeConnectivityMatrix
    from src.matrices.geocode_distance import GeocodeDistanceMatrix

    rows, cols = 4, 5
    n = rows * cols
    coords = np.array([[float(r), float(c)] for r in range(rows) for c in range(cols)])
    # scipy pdist order == GeocodeDistanceMatrix.condensed()
    condensed = pdist(coords)
    distance_matrix = GeocodeDistanceMatrix(condensed, coords)

    adj = np.zeros((n, n), dtype=np.int64)

    def node(r, c):
        return r * cols + c

    for r in range(rows):
        for c in range(cols):
            if r + 1 < rows:
                a, b = node(r, c), node(r + 1, c)
                adj[a, b] = adj[b, a] = 1
            if c + 1 < cols:
                a, b = node(r, c), node(r, c + 1)
                adj[a, b] = adj[b, a] = 1
    connectivity_matrix = GeocodeConnectivityMatrix(adj)
    geocodes = pl.Series("geocode", list(range(n)), dtype=pl.UInt64)
    return geocodes, condensed, distance_matrix, connectivity_matrix


def test_build_geocode_cluster_multi_k() -> None:
    print("src/dataframes/geocode_cluster.py  (build_geocode_cluster_multi_k):")
    from scipy.sparse import csr_matrix
    from sklearn.cluster import AgglomerativeClustering

    geocodes, condensed, distance_matrix, connectivity_matrix = _grid_cluster_fixture()
    n = len(geocodes)
    min_k, max_k = 2, 6

    edge_rows, edge_cols = np.nonzero(
        np.triu(connectivity_matrix._connectivity_matrix, k=1)
    )
    rust_df = bioregion_rs.build_geocode_cluster_multi_k(
        geocodes.to_list(),
        condensed.tolist(),
        edge_rows.tolist(),
        edge_cols.tolist(),
        min_k,
        max_k,
    )
    check(
        f"non-trivial: {rust_df.height} rows over {max_k - min_k + 1} k values",
        rust_df.height == n * (max_k - min_k + 1),
    )

    square = distance_matrix.squareform()
    all_match = True
    saw_multi_cluster = False
    for k in range(min_k, max_k + 1):
        # Reference: sklearn, exactly as the pre-cutover pipeline called it.
        ref = AgglomerativeClustering(
            n_clusters=k,
            connectivity=csr_matrix(connectivity_matrix._connectivity_matrix),
            linkage="ward",
        ).fit_predict(square)

        # geocodes are 0..n-1 in order, so sort("geocode") aligns with ref index.
        rust_k = (
            rust_df.filter(pl.col("num_clusters") == k)
            .sort("geocode")
            .get_column("cluster")
            .to_list()
        )
        if rust_k != ref.tolist():
            all_match = False
        if len(set(ref.tolist())) > 1:
            saw_multi_cluster = True

    check("cluster labels match sklearn exactly for every k", all_match)
    check("fixture actually produces multiple clusters", saw_multi_cluster)


def test_parquet_boundary() -> None:
    print("parquet boundary (stage handoff):")
    df = sample_frame()
    rust_out = bioregion_rs.select_geocode(df, 6)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "geocode.parquet"
        rust_out.write_parquet(p)
        reread = pl.scan_parquet(p).collect()
        py_out = select_geocode_lf(df.lazy(), geocode_precision=6).collect()
        check(
            "round-trips through parquet", reread["geocode"].equals(py_out["geocode"])
        )


def main() -> None:
    print(f"polars (python): {pl.__version__}\n")
    test_darken_hex_color()
    test_select_geocode()
    test_with_geocode()
    test_filter_by_bounding_box()
    test_build_taxonomy()
    test_build_geocode_taxa_counts()
    test_build_permanova_results()
    test_build_geocode_cluster_multi_k()
    test_parquet_boundary()
    print("\nAll interop + correctness checks passed.")


if __name__ == "__main__":
    main()
