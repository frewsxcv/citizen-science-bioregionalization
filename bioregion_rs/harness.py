"""Diff harness: prove the Rust <-> Python interop and correctness.

For each ported function, run the current Python implementation and the Rust
implementation on the same input and assert the outputs match. This is the
template every file migration follows.

Run:  uv run python bioregion_rs/harness.py
"""

import json
import math
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# Make the repo-root `src` package importable regardless of CWD.
sys.path.insert(0, str(REPO_ROOT))

import geojson as geojson_lib
import numpy as np
import polars as pl
import shapely
from scipy.spatial.distance import squareform

import bioregion_rs
from src.cluster_optimization import optimize_num_clusters
from src.colors import darken_hex_color as py_darken
from src.dataframes.cluster_boundary import build_cluster_boundary_df
from src.dataframes.cluster_color import build_cluster_color_df
from src.dataframes.cluster_neighbors import build_cluster_neighbors_df
from src.dataframes.cluster_significant_differences import (
    build_cluster_significant_differences_df,
)
from src.dataframes.cluster_taxa_statistics import build_cluster_taxa_statistics_df
from src.dataframes.darwin_core import build_darwin_core_lf
from src.dataframes.geocode import build_geocode_df
from src.dataframes.geocode_cluster_metrics import (
    build_geocode_cluster_metrics_df,
    select_optimal_k_elbow,
)
from src.dataframes.geocode_neighbors import (
    build_geocode_neighbors_df,
    build_geocode_neighbors_no_edges_df,
    graph as neighbors_graph,
)
from src.dataframes.geocode_silhouette_score import build_geocode_silhouette_score_df
from src.dataframes.geocode_taxa_counts import build_geocode_taxa_counts_lf
from src.dataframes.permanova_results import build_permanova_results_df
from src.dataframes.taxonomy import build_taxonomy_lf
from src.geocode import (
    filter_by_bounding_box as py_filter_bbox,
    select_geocode_lf,
    with_geocode_lf,
)
from src.geojson import build_geojson_feature_collection
from src.matrices.cluster_distance import ClusterDistanceMatrix
from src.matrices.geocode_connectivity import GeocodeConnectivityMatrix
from src.matrices.geocode_distance import GeocodeDistanceMatrix
from src.output import write_json_output
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
        check(f"precision={precision} is pl.DataFrame", isinstance(rust_out, pl.DataFrame))
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


# --- src/dataframes/geocode.py ----------------------------------------------


def _shapes(series: pl.Series) -> list:
    return [shapely.from_wkb(b) for b in series.to_list()]


def test_build_geocode() -> None:
    print("src/dataframes/geocode.py  (build_geocode: h3 -> center/boundary/is_edge):")
    # Geocode over all data (wide load bbox), but detect edges against a tighter
    # bbox that slices through the data, so both edge and interior hexagons occur.
    load_bbox = Bbox.from_coordinates(-90.0, 90.0, -180.0, 180.0)
    # max_lat=43.5 slices through several precision-4 cells (lat bounds straddle
    # it) while the lng~7 cluster stays interior -> both edge and non-edge cells.
    bbox = Bbox.from_coordinates(35.0, 43.5, -50.0, 10.0)
    precision = 4
    darwin = build_darwin_core_lf(str(REPO_ROOT / "test" / "sample-archive"), load_bbox)
    df = darwin.collect()

    rust_out = bioregion_rs.build_geocode(
        df, precision, bbox.min_lat, bbox.max_lat, bbox.min_lng, bbox.max_lng
    )
    py_out = build_geocode_df(darwin, precision, bbox)

    check(f"non-trivial: {py_out.height} geocodes built", py_out.height > 5)
    check("geocode set + order match", rust_out["geocode"].equals(py_out["geocode"]))
    check("is_edge matches exactly", rust_out["is_edge"].equals(py_out["is_edge"]))
    n_edges = int(py_out["is_edge"].sum())
    check(f"is_edge has both True and False ({n_edges} edges)", 0 < n_edges < py_out.height)

    # Geometry: compare decoded shapes. Centers by coordinate, boundaries by
    # symmetric-difference area (robust to vertex ordering / ULP differences
    # between h3o and polars_h3).
    rc, pc = _shapes(rust_out["center"]), _shapes(py_out["center"])
    rb, pb = _shapes(rust_out["boundary"]), _shapes(py_out["boundary"])
    max_center_err = max(abs(a.x - b.x) + abs(a.y - b.y) for a, b in zip(rc, pc))
    max_boundary_symdiff = max(
        a.symmetric_difference(b).area for a, b in zip(rb, pb)
    )
    check(f"centers match (max coord err {max_center_err:.2e})", max_center_err < 1e-7)
    check(
        f"boundaries match (max sym-diff area {max_boundary_symdiff:.2e})",
        max_boundary_symdiff < 1e-10,
    )


# --- src/dataframes/taxonomy.py, src/dataframes/geocode_taxa_counts.py -------


def _load_bbox_darwin_and_geocode_no_edges():
    """Shared fixture: sample-archive Darwin Core data + its non-edge geocodes."""
    load_bbox = Bbox.from_coordinates(-90.0, 90.0, -180.0, 180.0)
    bbox = Bbox.from_coordinates(35.0, 43.5, -50.0, 10.0)
    precision = 4
    darwin_lf = build_darwin_core_lf(str(REPO_ROOT / "test" / "sample-archive"), load_bbox)
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
        darwin_df, precision, geocode_no_edges_df,
        bbox.min_lat, bbox.max_lat, bbox.min_lng, bbox.max_lng,
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
        darwin_df, precision, geocode_no_edges_df,
        bbox.min_lat, bbox.max_lat, bbox.min_lng, bbox.max_lng,
    )
    py_taxonomy = build_taxonomy_lf(
        darwin_df.lazy(), precision, geocode_no_edges_df.lazy(), bbox
    ).collect()

    rust_counts = bioregion_rs.build_geocode_taxa_counts(
        darwin_df, precision, rust_taxonomy, geocode_no_edges_df,
        bbox.min_lat, bbox.max_lat, bbox.min_lng, bbox.max_lng,
    )
    py_counts = build_geocode_taxa_counts_lf(
        darwin_df.lazy(), precision, py_taxonomy.lazy(), geocode_no_edges_df.lazy(), bbox
    ).collect()

    check(f"non-trivial: {py_counts.height} (geocode, taxon) rows built", py_counts.height > 0)
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
            joined.select("geocode", "scientificName", "gbifTaxonId", "count").iter_rows()
        )

    check(
        "same (geocode, scientificName, gbifTaxonId, count) set",
        _resolved(rust_counts, rust_taxonomy) == _resolved(py_counts, py_taxonomy),
    )


# --- src/dataframes/geocode_neighbors.py --------------------------------------


def _neighbor_sets(df: pl.DataFrame, column: str, key: str = "geocode") -> dict:
    return {k: set(neighbors) for k, neighbors in df.select(key, column).iter_rows()}


def _is_single_component(df: pl.DataFrame, include_indirect_neighbors: bool) -> bool:
    import networkx as nx

    g = neighbors_graph(df, include_indirect_neighbors=include_indirect_neighbors)
    return nx.number_connected_components(g) == 1


def test_build_geocode_neighbors() -> None:
    print("src/dataframes/geocode_neighbors.py  (build_geocode_neighbors):")
    bbox, precision, _darwin_lf, darwin_df, _geocode_no_edges_df = (
        _load_bbox_darwin_and_geocode_no_edges()
    )
    geocode_df = build_geocode_df(darwin_df.lazy(), precision, bbox)

    rust_out = bioregion_rs.build_geocode_neighbors(geocode_df)
    py_out = build_geocode_neighbors_df(geocode_df)

    check(f"non-trivial: {py_out.height} geocodes", py_out.height > 1)
    check("same geocode order", rust_out["geocode"].equals(py_out["geocode"]))
    check(
        "same direct_neighbors set per geocode",
        _neighbor_sets(rust_out, "direct_neighbors") == _neighbor_sets(py_out, "direct_neighbors"),
    )
    check("rust output is a single connected component", _is_single_component(rust_out, True))
    check("python output is a single connected component", _is_single_component(py_out, True))
    check(
        "same direct_and_indirect_neighbors set per geocode",
        _neighbor_sets(rust_out, "direct_and_indirect_neighbors")
        == _neighbor_sets(py_out, "direct_and_indirect_neighbors"),
    )


def test_build_geocode_neighbors_no_edges() -> None:
    print("src/dataframes/geocode_neighbors.py  (build_geocode_neighbors_no_edges):")
    bbox, precision, _darwin_lf, darwin_df, geocode_no_edges_df = (
        _load_bbox_darwin_and_geocode_no_edges()
    )
    geocode_df = build_geocode_df(darwin_df.lazy(), precision, bbox)

    rust_neighbors = bioregion_rs.build_geocode_neighbors(geocode_df)
    py_neighbors = build_geocode_neighbors_df(geocode_df)

    rust_out = bioregion_rs.build_geocode_neighbors_no_edges(rust_neighbors, geocode_no_edges_df)
    py_out = build_geocode_neighbors_no_edges_df(py_neighbors, geocode_no_edges_df)

    check(f"non-trivial: {py_out.height} geocodes", 0 < py_out.height < geocode_df.height)
    check("sorted by geocode (rust)", rust_out["geocode"].is_sorted())
    check("same geocode set", set(rust_out["geocode"].to_list()) == set(py_out["geocode"].to_list()))
    check(
        "same direct_neighbors set per geocode",
        _neighbor_sets(rust_out, "direct_neighbors") == _neighbor_sets(py_out, "direct_neighbors"),
    )
    check("rust output is a single connected component", _is_single_component(rust_out, True))
    check("python output is a single connected component", _is_single_component(py_out, True))
    check(
        "same direct_and_indirect_neighbors set per geocode",
        _neighbor_sets(rust_out, "direct_and_indirect_neighbors")
        == _neighbor_sets(py_out, "direct_and_indirect_neighbors"),
    )


# --- src/matrices/geocode_connectivity.py -------------------------------------


def test_build_geocode_connectivity_matrix() -> None:
    print("src/matrices/geocode_connectivity.py  (build_geocode_connectivity_matrix):")
    bbox, precision, _darwin_lf, darwin_df, _geocode_no_edges_df = (
        _load_bbox_darwin_and_geocode_no_edges()
    )
    geocode_df = build_geocode_df(darwin_df.lazy(), precision, bbox)
    geocode_neighbors_df = build_geocode_neighbors_df(geocode_df)

    rust_matrix = bioregion_rs.build_geocode_connectivity_matrix(geocode_neighbors_df)
    py_matrix = GeocodeConnectivityMatrix.build(geocode_neighbors_df)._connectivity_matrix

    check(f"non-trivial: {len(rust_matrix)}x{len(rust_matrix)} matrix", len(rust_matrix) > 1)
    check(
        "matrix matches exactly",
        np.array(rust_matrix, dtype=int).tolist() == py_matrix.astype(int).tolist(),
    )


# --- src/dataframes/cluster_taxa_statistics.py --------------------------------


def test_build_cluster_taxa_statistics() -> None:
    print("src/dataframes/cluster_taxa_statistics.py  (build_cluster_taxa_statistics):")
    bbox, precision, _darwin_lf, darwin_df, geocode_no_edges_df = (
        _load_bbox_darwin_and_geocode_no_edges()
    )
    py_taxonomy = build_taxonomy_lf(
        darwin_df.lazy(), precision, geocode_no_edges_df.lazy(), bbox
    ).collect()
    py_counts = build_geocode_taxa_counts_lf(
        darwin_df.lazy(), precision, py_taxonomy.lazy(), geocode_no_edges_df.lazy(), bbox
    ).collect()

    # Synthetic cluster assignment (real clustering is sklearn Ward
    # agglomerative clustering, out of scope for this port): split geocodes
    # into two clusters deterministically so both the overall and per-cluster
    # aggregation paths get exercised.
    # Cluster by row index (not geocode value) since H3 cell IDs at a given
    # resolution can share low-order bits, making `geocode % k` degenerate.
    geocodes = py_counts["geocode"].unique().sort()
    geocode_cluster_df = pl.DataFrame(
        {"geocode": geocodes, "cluster": [i % 2 for i in range(len(geocodes))]}
    ).cast({"geocode": pl.UInt64, "cluster": pl.UInt32})

    rust_out = bioregion_rs.build_cluster_taxa_statistics(
        py_counts, geocode_cluster_df, py_taxonomy
    )
    py_out = build_cluster_taxa_statistics_df(
        py_counts.lazy(), geocode_cluster_df.lazy(), py_taxonomy.lazy()
    )

    check(f"non-trivial: {py_out.height} rows", py_out.height > 0)

    def _rows(df: pl.DataFrame) -> set:
        return {
            (cluster, taxon_id, count, round(average, 9))
            for cluster, taxon_id, count, average in df.select(
                "cluster", "taxonId", "count", "average"
            ).iter_rows()
        }

    check("same (cluster, taxonId, count, average) row set", _rows(rust_out) == _rows(py_out))


# --- src/dataframes/cluster_neighbors.py --------------------------------------


def test_build_cluster_neighbors() -> None:
    print("src/dataframes/cluster_neighbors.py  (build_cluster_neighbors):")
    bbox, precision, _darwin_lf, darwin_df, geocode_no_edges_df = (
        _load_bbox_darwin_and_geocode_no_edges()
    )
    geocode_df = build_geocode_df(darwin_df.lazy(), precision, bbox)
    py_neighbors = build_geocode_neighbors_df(geocode_df)
    py_neighbors_no_edges = build_geocode_neighbors_no_edges_df(
        py_neighbors, geocode_no_edges_df
    )

    # Cluster by row index (not geocode value) since H3 cell IDs at a given
    # resolution can share low-order bits, making `geocode % k` degenerate.
    # Real clustering is sklearn Ward, out of scope here; 3 clusters so
    # cross-cluster adjacency actually gets exercised.
    geocodes = py_neighbors_no_edges["geocode"].sort()
    geocode_cluster_df = pl.DataFrame(
        {"geocode": geocodes, "cluster": [i % 3 for i in range(len(geocodes))]}
    ).cast({"geocode": pl.UInt64, "cluster": pl.UInt32})

    rust_out = bioregion_rs.build_cluster_neighbors(py_neighbors_no_edges, geocode_cluster_df)
    py_out = build_cluster_neighbors_df(py_neighbors_no_edges, geocode_cluster_df)

    check(f"non-trivial: {py_out.height} clusters", py_out.height > 1)
    check(
        "same direct_neighbors set per cluster",
        _neighbor_sets(rust_out, "direct_neighbors", key="cluster")
        == _neighbor_sets(py_out, "direct_neighbors", key="cluster"),
    )
    check(
        "same direct_and_indirect_neighbors set per cluster",
        _neighbor_sets(rust_out, "direct_and_indirect_neighbors", key="cluster")
        == _neighbor_sets(py_out, "direct_and_indirect_neighbors", key="cluster"),
    )


# --- src/matrices/cluster_distance.py -----------------------------------------


def _pairwise_distances(condensed, cluster_ids) -> dict:
    square = squareform(np.asarray(condensed))
    return {
        tuple(sorted((c1, c2))): round(float(square[i][j]), 9)
        for i, c1 in enumerate(cluster_ids)
        for j, c2 in enumerate(cluster_ids)
        if i < j
    }


def test_build_cluster_distance_matrix() -> None:
    print("src/matrices/cluster_distance.py  (build_cluster_distance_matrix):")
    bbox, precision, _darwin_lf, darwin_df, geocode_no_edges_df = (
        _load_bbox_darwin_and_geocode_no_edges()
    )
    py_taxonomy = build_taxonomy_lf(
        darwin_df.lazy(), precision, geocode_no_edges_df.lazy(), bbox
    ).collect()
    py_counts = build_geocode_taxa_counts_lf(
        darwin_df.lazy(), precision, py_taxonomy.lazy(), geocode_no_edges_df.lazy(), bbox
    ).collect()

    # Cluster by row index (not geocode value) since H3 cell IDs at a given
    # resolution can share low-order bits, making `geocode % k` degenerate.
    # Real clustering is sklearn Ward, out of scope here. Use one cluster per
    # geocode (not just 2): with exactly 2 clusters, RobustScaler always
    # scales every non-constant column to a perfect +-1 pair, so Bray-Curtis's
    # sum(|u+v|) denominator is ~0 for every such column and the distance
    # becomes a huge, rounding-dominated number that two independent
    # implementations have no reason to agree on bit-for-bit.
    geocodes = py_counts["geocode"].unique().sort()
    n = len(geocodes)
    geocode_cluster_df = pl.DataFrame(
        {"geocode": geocodes, "cluster": list(range(n))}
    ).cast({"geocode": pl.UInt64, "cluster": pl.UInt32})
    cluster_taxa_stats = build_cluster_taxa_statistics_df(
        py_counts.lazy(), geocode_cluster_df.lazy(), py_taxonomy.lazy()
    )

    rust_condensed, rust_cluster_ids = bioregion_rs.build_cluster_distance_matrix(
        cluster_taxa_stats
    )
    py_matrix = ClusterDistanceMatrix.build(cluster_taxa_stats)

    check(f"non-trivial: {len(rust_cluster_ids)} clusters", len(rust_cluster_ids) > 1)

    rust_distances = _pairwise_distances(rust_condensed, rust_cluster_ids)
    py_distances = _pairwise_distances(py_matrix.condensed(), py_matrix.cluster_ids())

    check("same cluster ID pairs", set(rust_distances) == set(py_distances))
    check(
        "same pairwise Bray-Curtis distances",
        all(abs(rust_distances[k] - py_distances[k]) < 1e-9 for k in rust_distances),
    )


# --- src/dataframes/cluster_color.py (geographic path only) ------------------


def test_build_cluster_color() -> None:
    print("src/dataframes/cluster_color.py  (build_cluster_color, geographic):")
    bbox, precision, _darwin_lf, darwin_df, geocode_no_edges_df = (
        _load_bbox_darwin_and_geocode_no_edges()
    )
    geocode_df = build_geocode_df(darwin_df.lazy(), precision, bbox)
    py_neighbors = build_geocode_neighbors_df(geocode_df)
    py_neighbors_no_edges = build_geocode_neighbors_no_edges_df(
        py_neighbors, geocode_no_edges_df
    )

    # Cluster by row index (not geocode value): see the note in
    # test_build_cluster_neighbors. Use enough clusters that the adjacency
    # graph actually needs more than one color.
    geocodes = py_neighbors_no_edges["geocode"].sort()
    geocode_cluster_df = pl.DataFrame(
        {"geocode": geocodes, "cluster": [i % 4 for i in range(len(geocodes))]}
    ).cast({"geocode": pl.UInt64, "cluster": pl.UInt32})
    cluster_neighbors_df = build_cluster_neighbors_df(
        py_neighbors_no_edges, geocode_cluster_df
    )

    rust_out = bioregion_rs.build_cluster_color(cluster_neighbors_df)
    py_out = build_cluster_color_df(
        cluster_neighbors_df.lazy(), color_method="geographic"
    )

    check(f"non-trivial: {py_out.height} clusters colored", py_out.height > 1)

    def _rows(df: pl.DataFrame) -> set:
        return set(df.select("cluster", "color", "darkened_color").iter_rows())

    check("same (cluster, color, darkened_color) row set", _rows(rust_out) == _rows(py_out))


# --- src/dataframes/cluster_boundary.py ---------------------------------------


def test_build_cluster_boundary() -> None:
    print("src/dataframes/cluster_boundary.py  (build_cluster_boundary):")
    _bbox, _precision, _darwin_lf, _darwin_df, geocode_no_edges_df = (
        _load_bbox_darwin_and_geocode_no_edges()
    )

    # Cluster by row index (not geocode value): see the note in
    # test_build_cluster_neighbors.
    geocodes = geocode_no_edges_df["geocode"].sort()
    geocode_cluster_df = pl.DataFrame(
        {"geocode": geocodes, "cluster": [i % 3 for i in range(len(geocodes))]}
    ).cast({"geocode": pl.UInt64, "cluster": pl.UInt32})

    rust_out = bioregion_rs.build_cluster_boundary(geocode_cluster_df, geocode_no_edges_df)
    py_out = build_cluster_boundary_df(geocode_cluster_df, geocode_no_edges_df.lazy())

    check(f"non-trivial: {py_out.height} cluster boundaries", py_out.height > 1)
    check(
        "same cluster set",
        set(rust_out["cluster"].to_list()) == set(py_out["cluster"].to_list()),
    )

    # Relative (not absolute) tolerance: geo's i_overlay-based unary_union and
    # GEOS's union algorithm are different implementations, so tiny
    # floating-point differences in how they resolve shared hexagon edges are
    # expected (unlike the direct WKB pass-through checks elsewhere in this
    # harness) — this is the "geometry robustness" caveat the migration plan
    # already flagged, not a correctness bug.
    rust_geoms = dict(rust_out.select("cluster", "geometry").iter_rows())
    py_geoms = dict(py_out.select("cluster", "geometry").iter_rows())
    max_relative_symdiff = max(
        shapely.from_wkb(rust_geoms[cluster])
        .symmetric_difference(shapely.from_wkb(py_geoms[cluster]))
        .area
        / shapely.from_wkb(py_geoms[cluster]).area
        for cluster in rust_geoms
    )
    check(
        f"geometries match (max relative sym-diff {max_relative_symdiff:.2e})",
        max_relative_symdiff < 1e-5,
    )


# --- src/dataframes/cluster_significant_differences.py ------------------------


def test_build_cluster_significant_differences() -> None:
    print(
        "src/dataframes/cluster_significant_differences.py  "
        "(build_cluster_significant_differences):"
    )
    # Hand-built (not derived from the sample archive): its real taxa counts
    # are all well under MIN_COUNT_THRESHOLD=5, so nothing would ever reach
    # the Fisher's-exact/scoring logic this is meant to exercise.
    all_stats = pl.DataFrame(
        {
            "cluster": [0, 0, 1, 1, 2, 2],
            "taxonId": [1, 2, 1, 2, 1, 2],
            "count": [50, 10, 5, 55, 30, 30],
            "average": [0.0] * 6,
        }
    ).cast(
        {
            "cluster": pl.UInt32,
            "taxonId": pl.UInt32,
            "count": pl.UInt32,
            "average": pl.Float64,
        }
    )
    cluster_neighbors_df = pl.DataFrame(
        {
            "cluster": [0, 1, 2],
            "direct_neighbors": [[1], [0, 2], [1]],
            "direct_and_indirect_neighbors": [[1], [0, 2], [1]],
        }
    ).cast(
        {
            "cluster": pl.UInt32,
            "direct_neighbors": pl.List(pl.UInt32),
            "direct_and_indirect_neighbors": pl.List(pl.UInt32),
        }
    )

    rust_out = bioregion_rs.build_cluster_significant_differences(all_stats, cluster_neighbors_df)
    py_out = build_cluster_significant_differences_df(all_stats, cluster_neighbors_df.lazy())

    check(f"non-trivial: {py_out.height} significant differences found", py_out.height > 0)

    def _rows(df: pl.DataFrame) -> set:
        return {
            (cluster, taxon_id, round(p, 9), round(log2fc, 9), cc, nc, round(hs, 9), round(ls, 9))
            for cluster, taxon_id, p, log2fc, cc, nc, hs, ls in df.select(
                "cluster",
                "taxonId",
                "p_value",
                "log2_fold_change",
                "cluster_count",
                "neighbor_count",
                "high_log2_high_count_score",
                "low_log2_high_count_score",
            ).iter_rows()
        }

    check("same rows (all columns)", _rows(rust_out) == _rows(py_out))


# --- src/dataframes/permanova_results.py --------------------------------------


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
    geocode_lf = pl.DataFrame({"geocode": geocode_ids}).cast({"geocode": pl.UInt64}).lazy()

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


# --- src/dataframes/geocode_cluster_metrics.py --------------------------------


def _six_point_three_pair_fixture():
    """Hand-built (not sample-archive-derived): 6 points forming 3 tight
    pairs, each pair far from the others, tested at both k=2 and k=3."""
    geocode_ids = [1, 2, 3, 4, 5, 6]
    condensed = [
        0.1, 2.0, 2.0, 3.0, 3.0,  # (1,2) (1,3) (1,4) (1,5) (1,6)
             2.0, 2.0, 3.0, 3.0,  # (2,3) (2,4) (2,5) (2,6)
                  0.1, 3.0, 3.0,  # (3,4) (3,5) (3,6)
                       3.0, 3.0,  # (4,5) (4,6)
                            0.1,  # (5,6)
    ]  # fmt: skip
    geocode_cluster_multi_k_df = pl.DataFrame(
        {
            "geocode": geocode_ids + geocode_ids,
            "cluster": [0, 0, 1, 1, 2, 2] + [0, 0, 0, 1, 1, 1],
            "num_clusters": [3] * 6 + [2] * 6,
        }
    ).cast({"geocode": pl.UInt64, "cluster": pl.UInt32, "num_clusters": pl.UInt32})
    distance_matrix = GeocodeDistanceMatrix(
        condensed=np.array(condensed), reduced_features=np.empty((6, 0))
    )
    return condensed, geocode_cluster_multi_k_df, distance_matrix


# --- src/dataframes/geocode_cluster.py ----------------------------------------


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


def test_build_geocode_cluster_metrics() -> None:
    print(
        "src/dataframes/geocode_cluster_metrics.py  (build_geocode_cluster_metrics):"
    )
    condensed, geocode_cluster_multi_k_df, distance_matrix = (
        _six_point_three_pair_fixture()
    )

    rust_out = bioregion_rs.build_geocode_cluster_metrics(
        condensed, geocode_cluster_multi_k_df
    )
    py_out = build_geocode_cluster_metrics_df(distance_matrix, geocode_cluster_multi_k_df)

    check(f"non-trivial: {py_out.height} k values tested", py_out.height == 2)

    def _rows(df: pl.DataFrame) -> set:
        return {
            (
                row["num_clusters"],
                round(row["silhouette_score"], 6),
                round(row["calinski_harabasz_score"], 6),
                round(row["davies_bouldin_score"], 6),
                round(row["inertia"], 6),
                round(row["combined_score"], 6),
            )
            for row in df.iter_rows(named=True)
        }

    check("same metrics per k (all engines)", _rows(rust_out) == _rows(py_out))

    # Kneedle elbow selection: select_optimal_k_elbow only ever touches the
    # num_clusters/inertia columns, so a minimal 2-column DataFrame suffices
    # (it isn't run through GeocodeClusterMetricsSchema.validate()).
    k_values = [2, 3, 4, 5, 6]
    inertia_values = [100.0, 50.0, 20.0, 18.0, 17.0]
    metrics_df = pl.DataFrame(
        {"num_clusters": k_values, "inertia": inertia_values}
    ).cast({"num_clusters": pl.UInt32, "inertia": pl.Float64})

    rust_k = bioregion_rs.select_optimal_k_elbow(k_values, inertia_values, 1.0)
    py_k = select_optimal_k_elbow(metrics_df, sensitivity=1.0)
    check(f"elbow k matches (rust={rust_k}, py={py_k})", rust_k == py_k)


# --- src/dataframes/geocode_silhouette_score.py -------------------------------


def test_build_geocode_silhouette_score() -> None:
    print(
        "src/dataframes/geocode_silhouette_score.py  (build_geocode_silhouette_score):"
    )
    condensed, geocode_cluster_multi_k_df, distance_matrix = (
        _six_point_three_pair_fixture()
    )

    rust_out = bioregion_rs.build_geocode_silhouette_score(
        condensed, geocode_cluster_multi_k_df
    )
    py_out = build_geocode_silhouette_score_df(distance_matrix, geocode_cluster_multi_k_df)

    check(f"non-trivial: {py_out.height} rows", py_out.height == 2 * (6 + 1))

    def _rows(df: pl.DataFrame) -> set:
        return {
            (row["geocode"], row["num_clusters"], round(row["silhouette_score"], 9))
            for row in df.iter_rows(named=True)
        }

    check("same (geocode, num_clusters, silhouette_score) rows", _rows(rust_out) == _rows(py_out))


# --- src/cluster_optimization.py -----------------------------------------------


def test_optimize_num_clusters() -> None:
    print("src/cluster_optimization.py  (optimize_num_clusters):")
    # Only 2 distinct k values (2, 3) are tested here, which is below
    # _find_elbow_point's own `len(df) < 3` minimum -- so this specifically
    # exercises the "no elbow found, fall back to highest combined_score"
    # path. Elbow-found behavior itself is already covered by
    # geocode_cluster_metrics.py's own tests; this file is pure orchestration
    # on top of that.
    condensed, geocode_cluster_multi_k_df, distance_matrix = (
        _six_point_three_pair_fixture()
    )

    rust_k, rust_metrics = bioregion_rs.optimize_num_clusters(
        condensed, geocode_cluster_multi_k_df
    )
    py_k, py_metrics = optimize_num_clusters(distance_matrix, geocode_cluster_multi_k_df)

    check(f"same optimal k (rust={rust_k}, py={py_k})", rust_k == py_k)

    def _rows(df: pl.DataFrame) -> set:
        return {
            (row["num_clusters"], round(row["combined_score"], 6))
            for row in df.iter_rows(named=True)
        }

    check("same metrics table (combined_score by k)", _rows(rust_metrics) == _rows(py_metrics))


# --- src/geojson.py -------------------------------------------------------------


def test_build_geojson_feature_collection() -> None:
    print("src/geojson.py  (build_geojson_feature_collection):")
    bbox, precision, _darwin_lf, darwin_df, geocode_no_edges_df = (
        _load_bbox_darwin_and_geocode_no_edges()
    )
    geocode_df = build_geocode_df(darwin_df.lazy(), precision, bbox)

    # Cluster by row index (not geocode value): see the note in
    # test_build_cluster_neighbors. 3 clusters -> a mix of single-geocode
    # (bare Polygon) and multi-geocode (MultiPolygon) cluster boundaries,
    # exercising both geometry shapes build_geojson_feature_collection emits.
    geocodes = geocode_no_edges_df["geocode"].sort()
    geocode_cluster_df = pl.DataFrame(
        {"geocode": geocodes, "cluster": [i % 3 for i in range(len(geocodes))]}
    ).cast({"geocode": pl.UInt64, "cluster": pl.UInt32})

    cluster_boundary_df = build_cluster_boundary_df(
        geocode_cluster_df, geocode_no_edges_df.lazy()
    )

    py_neighbors = build_geocode_neighbors_df(geocode_df)
    py_neighbors_no_edges = build_geocode_neighbors_no_edges_df(
        py_neighbors, geocode_no_edges_df
    )
    cluster_neighbors_df = build_cluster_neighbors_df(
        py_neighbors_no_edges, geocode_cluster_df
    )
    cluster_color_df = build_cluster_color_df(
        cluster_neighbors_df.lazy(), color_method="geographic"
    )

    rust_json = bioregion_rs.build_geojson_feature_collection(
        cluster_boundary_df, cluster_color_df
    )
    py_feature_collection = build_geojson_feature_collection(
        cluster_boundary_df, cluster_color_df
    )

    rust_parsed = json.loads(rust_json)
    py_parsed = json.loads(geojson_lib.dumps(py_feature_collection))

    check(f"non-trivial: {len(py_parsed['features'])} features", len(py_parsed["features"]) > 1)
    check(
        "both are FeatureCollections",
        rust_parsed["type"] == "FeatureCollection" and py_parsed["type"] == "FeatureCollection",
    )

    def _by_cluster(fc: dict) -> dict:
        return {f["properties"]["cluster"]: f for f in fc["features"]}

    rust_by_cluster = _by_cluster(rust_parsed)
    py_by_cluster = _by_cluster(py_parsed)
    check("same cluster set", set(rust_by_cluster) == set(py_by_cluster))
    check(
        "same properties per cluster",
        all(
            rust_by_cluster[c]["properties"] == py_by_cluster[c]["properties"]
            for c in rust_by_cluster
        ),
    )

    # Geometry: relative tolerance, not exact equality -- these boundaries
    # come from geo::unary_union (see cluster_boundary.rs), which is a
    # different implementation from GEOS's union and differs by a tiny
    # (~1e-7 relative) amount at shared hexagon edges, same as
    # test_build_cluster_boundary.
    max_relative_symdiff = max(
        shapely.geometry.shape(rust_by_cluster[c]["geometry"])
        .symmetric_difference(shapely.geometry.shape(py_by_cluster[c]["geometry"]))
        .area
        / shapely.geometry.shape(py_by_cluster[c]["geometry"]).area
        for c in rust_by_cluster
    )
    check(
        f"geometries match (max relative sym-diff {max_relative_symdiff:.2e})",
        max_relative_symdiff < 1e-5,
    )


def test_write_json_output() -> None:
    print("src/output.py  (write_json_output):")

    # Reuses test_build_cluster_significant_differences's fixture (its real
    # taxa counts clear MIN_COUNT_THRESHOLD=5 and produce a non-trivial
    # result set to build the rest of this test's fixtures on top of).
    all_stats = pl.DataFrame(
        {
            "cluster": [0, 0, 1, 1, 2, 2],
            "taxonId": [1, 2, 1, 2, 1, 2],
            "count": [50, 10, 5, 55, 30, 30],
            "average": [0.0] * 6,
        }
    ).cast(
        {
            "cluster": pl.UInt32,
            "taxonId": pl.UInt32,
            "count": pl.UInt32,
            "average": pl.Float64,
        }
    )
    cluster_neighbors_df = pl.DataFrame(
        {
            "cluster": [0, 1, 2],
            "direct_neighbors": [[1], [0, 2], [1]],
            "direct_and_indirect_neighbors": [[1], [0, 2], [1]],
        }
    ).cast(
        {
            "cluster": pl.UInt32,
            "direct_neighbors": pl.List(pl.UInt32),
            "direct_and_indirect_neighbors": pl.List(pl.UInt32),
        }
    )
    cluster_significant_differences_df = build_cluster_significant_differences_df(
        all_stats, cluster_neighbors_df.lazy()
    )
    check(
        f"non-trivial: {cluster_significant_differences_df.height} significant differences",
        cluster_significant_differences_df.height > 0,
    )

    clusters_present = (
        cluster_significant_differences_df["cluster"].unique().sort().to_list()
    )
    cluster_boundary_df = pl.DataFrame(
        {
            "cluster": clusters_present,
            "geometry": [
                shapely.to_wkb(shapely.geometry.box(i, i, i + 1, i + 1))
                for i in range(len(clusters_present))
            ],
        }
    ).cast({"cluster": pl.UInt32, "geometry": pl.Binary})
    cluster_color_df = pl.DataFrame(
        {
            "cluster": clusters_present,
            "color": [f"#{i:02x}{i:02x}{i:02x}" for i in range(len(clusters_present))],
            "darkened_color": [
                py_darken(f"#{i:02x}{i:02x}{i:02x}") for i in range(len(clusters_present))
            ],
        }
    ).cast({"cluster": pl.UInt32, "color": pl.String, "darkened_color": pl.String})

    # Exercise both join edge cases write_json_output relies on: a taxonId
    # present in the significant differences but absent from taxonomy_df is
    # dropped (inner join), and a taxonId present in taxonomy_df but absent
    # from significant_taxa_images_df gets a null image_url (left join).
    present_taxon_ids = (
        cluster_significant_differences_df["taxonId"].unique().sort().to_list()
    )
    taxonomy_taxon_ids = (
        present_taxon_ids[:-1] if len(present_taxon_ids) > 1 else present_taxon_ids
    )
    taxonomy_df = pl.DataFrame(
        {
            "taxonId": taxonomy_taxon_ids,
            "scientificName": [f"Taxon {t}" for t in taxonomy_taxon_ids],
            "gbifTaxonId": [1000 + t for t in taxonomy_taxon_ids],
        }
    ).cast(
        {"taxonId": pl.UInt32, "scientificName": pl.String, "gbifTaxonId": pl.UInt32}
    )
    images_taxon_ids = (
        taxonomy_taxon_ids[:-1] if len(taxonomy_taxon_ids) > 1 else []
    )
    significant_taxa_images_df = pl.DataFrame(
        {
            "taxonId": images_taxon_ids,
            "image_url": [f"https://example.com/{t}.jpg" for t in images_taxon_ids],
        }
    ).cast({"taxonId": pl.UInt32, "image_url": pl.String})

    rust_json = bioregion_rs.build_json_output(
        cluster_significant_differences_df,
        cluster_boundary_df,
        taxonomy_df,
        cluster_color_df,
        significant_taxa_images_df,
    )
    rust_parsed = json.loads(rust_json)

    with tempfile.TemporaryDirectory() as d:
        out_path = str(Path(d) / "output.json")
        write_json_output(
            cluster_significant_differences_df,
            cluster_boundary_df,
            taxonomy_df,
            cluster_color_df,
            significant_taxa_images_df,
            out_path,
        )
        with open(out_path) as f:
            py_parsed = json.load(f)

    check(f"non-trivial: {len(py_parsed)} clusters", len(py_parsed) > 1)
    check("dropped a taxonId missing from taxonomy_df", len(taxonomy_taxon_ids) < len(present_taxon_ids))
    check("left a taxonId's image_url null", len(images_taxon_ids) < len(taxonomy_taxon_ids))

    def _by_cluster(entries: list) -> dict:
        return {e["cluster"]: e for e in entries}

    rust_by_cluster = _by_cluster(rust_parsed)
    py_by_cluster = _by_cluster(py_parsed)
    check("same cluster set", set(rust_by_cluster) == set(py_by_cluster))
    check(
        "same color/darkened_color per cluster",
        all(
            rust_by_cluster[c]["color"] == py_by_cluster[c]["color"]
            and rust_by_cluster[c]["darkened_color"] == py_by_cluster[c]["darkened_color"]
            for c in rust_by_cluster
        ),
    )
    check(
        "same boundary geometry per cluster",
        all(
            rust_by_cluster[c]["boundary"] == py_by_cluster[c]["boundary"]
            for c in rust_by_cluster
        ),
    )

    def _taxa_set(entry: dict) -> set:
        # Order isn't guaranteed by either implementation's join, so compare
        # as a set of rows rather than an exact list.
        return {tuple(sorted(t.items())) for t in entry["significant_taxa"]}

    check(
        "same significant_taxa per cluster (order-independent)",
        all(
            _taxa_set(rust_by_cluster[c]) == _taxa_set(py_by_cluster[c])
            for c in rust_by_cluster
        ),
    )


# --- parquet stage boundary --------------------------------------------------


def test_parquet_boundary() -> None:
    print("parquet boundary (stage handoff):")
    df = sample_frame()
    rust_out = bioregion_rs.select_geocode(df, 6)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "geocode.parquet"
        rust_out.write_parquet(p)
        reread = pl.scan_parquet(p).collect()
        py_out = select_geocode_lf(df.lazy(), geocode_precision=6).collect()
        check("round-trips through parquet", reread["geocode"].equals(py_out["geocode"]))


def main() -> None:
    print(f"polars (python): {pl.__version__}\n")
    test_darken_hex_color()
    test_select_geocode()
    test_with_geocode()
    test_filter_by_bounding_box()
    test_build_geocode()
    test_build_taxonomy()
    test_build_geocode_taxa_counts()
    test_build_geocode_neighbors()
    test_build_geocode_neighbors_no_edges()
    test_build_geocode_connectivity_matrix()
    test_build_cluster_taxa_statistics()
    test_build_cluster_neighbors()
    test_build_cluster_distance_matrix()
    test_build_cluster_color()
    test_build_cluster_boundary()
    test_build_cluster_significant_differences()
    test_build_permanova_results()
    test_build_geocode_cluster_multi_k()
    test_build_geocode_cluster_metrics()
    test_build_geocode_silhouette_score()
    test_optimize_num_clusters()
    test_build_geojson_feature_collection()
    test_write_json_output()
    test_parquet_boundary()
    print("\nAll interop + correctness checks passed.")


if __name__ == "__main__":
    main()
