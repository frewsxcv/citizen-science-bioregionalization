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

import polars as pl
import shapely

import bioregion_rs
from src.colors import darken_hex_color as py_darken
from src.dataframes.darwin_core import build_darwin_core_lf
from src.dataframes.geocode import build_geocode_df
from src.geocode import (
    filter_by_bounding_box as py_filter_bbox,
    select_geocode_lf,
    with_geocode_lf,
)
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
    test_parquet_boundary()
    print("\nAll interop + correctness checks passed.")


if __name__ == "__main__":
    main()
