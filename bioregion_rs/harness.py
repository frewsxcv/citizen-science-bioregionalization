"""Diff harness: prove the Rust <-> Python interop and correctness.

For each ported function, run the current Python implementation and the Rust
implementation on the same input and assert the outputs match. This is the
template every file migration follows.

Run:  uv run python bioregion_rs/harness.py
"""

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# Make the repo-root `src` package importable regardless of CWD.
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import polars as pl
import shapely

import bioregion_rs
from src.colors import darken_hex_color as py_darken
from src.dataframes.cluster_neighbors import build_cluster_neighbors_df
from src.dataframes.cluster_taxa_statistics import build_cluster_taxa_statistics_df
from src.dataframes.darwin_core import build_darwin_core_lf
from src.dataframes.geocode import build_geocode_df
from src.dataframes.geocode_neighbors import (
    build_geocode_neighbors_df,
    build_geocode_neighbors_no_edges_df,
    graph as neighbors_graph,
)
from src.dataframes.geocode_taxa_counts import build_geocode_taxa_counts_lf
from src.dataframes.taxonomy import build_taxonomy_lf
from src.geocode import (
    filter_by_bounding_box as py_filter_bbox,
    select_geocode_lf,
    with_geocode_lf,
)
from src.matrices.geocode_connectivity import GeocodeConnectivityMatrix
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
    geocodes = py_counts["geocode"].unique().sort()
    geocode_cluster_df = pl.DataFrame(
        {"geocode": geocodes, "cluster": [g % 2 for g in geocodes]}
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

    # Synthetic cluster assignment covering every geocode in the no-edges
    # neighbor set (real clustering is sklearn Ward, out of scope here): split
    # into 3 clusters deterministically so cross-cluster adjacency actually
    # gets exercised.
    geocodes = py_neighbors_no_edges["geocode"].sort()
    geocode_cluster_df = pl.DataFrame(
        {"geocode": geocodes, "cluster": [g % 3 for g in geocodes]}
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
    test_parquet_boundary()
    print("\nAll interop + correctness checks passed.")


if __name__ == "__main__":
    main()
