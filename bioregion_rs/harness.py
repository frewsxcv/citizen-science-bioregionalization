"""Phase 0 diff harness: prove the Rust <-> Python interop and correctness.

For each ported function, run the current Python implementation and the Rust
implementation on the same input and assert the outputs match. This is the
template every subsequent file migration will follow.

Run:  uv run python bioregion_rs/harness.py
"""

import sys
import tempfile
from pathlib import Path

# Make the repo-root `src` package importable regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import polars as pl

import bioregion_rs
from src.colors import darken_hex_color as py_darken
from src.geocode import select_geocode_lf


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


def test_pure_pyo3_boundary() -> None:
    """Plain PyO3: pure function, no polars."""
    print("pure PyO3 boundary (darken_hex_color):")
    cases = ["#ff0000", "#00ff00", "#123456", "#f0a", "#ffffff", "#000000"]
    for factor in (0.5, 0.3, 0.75, 1.0):
        for c in cases:
            check(
                f"{c} @ {factor}",
                bioregion_rs.darken_hex_color(c, factor) == py_darken(c, factor),
            )


def test_pyo3_polars_dataframe_boundary() -> None:
    """pyo3-polars: pl.DataFrame in, pl.DataFrame out (Arrow FFI, zero-copy)."""
    print("pyo3-polars DataFrame boundary (select_geocode):")
    df = sample_frame()
    for precision in (2, 4, 6, 8):
        rust_out = bioregion_rs.select_geocode(df, precision)
        py_out = select_geocode_lf(df.lazy(), geocode_precision=precision).collect()
        check(
            f"precision={precision} type is pl.DataFrame",
            isinstance(rust_out, pl.DataFrame),
        )
        check(
            f"precision={precision} geocode column matches polars_h3",
            rust_out["geocode"].equals(py_out["geocode"]),
        )


def test_parquet_boundary() -> None:
    """The migration seam: Rust writes parquet, Python reads it (and vice versa)."""
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
    test_pure_pyo3_boundary()
    test_pyo3_polars_dataframe_boundary()
    test_parquet_boundary()
    print("\nAll Phase 0 interop checks passed.")


if __name__ == "__main__":
    main()
