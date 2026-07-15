//! `bioregion_rs` — Rust core for the bioregionalization pipeline.
//!
//! Phase 0 scaffold: proves the Rust <-> Python interop paths described in
//! `RUST_MIGRATION_PLAN.md`.
//!
//! - `darken_hex_color`: a pure function, proving the plain PyO3 boundary.
//! - `select_geocode`: a `pl.DataFrame` in / `pl.DataFrame` out function using
//!   `h3o`, proving the zero-copy `pyo3-polars` DataFrame boundary + a real
//!   Phase 1 leaf (mirrors `src/geocode.py::select_geocode_lf`).

use h3o::{LatLng, Resolution};
use polars::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

/// Convert a polars error into a Python `ValueError`.
fn to_py(e: PolarsError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Parse an H3 resolution (0-15) from a raw integer.
fn resolution_from_u8(precision: u8) -> PolarsResult<Resolution> {
    Resolution::try_from(precision).map_err(|e| {
        PolarsError::ComputeError(format!("invalid H3 resolution {precision}: {e}").into())
    })
}

/// Map paired lat/lng f64 series to a `geocode` UInt64 series of H3 cell ids.
/// Null in either coordinate, or an invalid coordinate, yields a null geocode.
fn latlng_to_geocode(lat: &Series, lng: &Series, res: Resolution) -> PolarsResult<Series> {
    let lat = lat.f64()?;
    let lng = lng.f64()?;
    let geocode: UInt64Chunked = lat
        .iter()
        .zip(lng.iter())
        .map(|(la, lo)| match (la, lo) {
            (Some(la), Some(lo)) => {
                LatLng::new(la, lo).ok().map(|ll| u64::from(ll.to_cell(res)))
            }
            _ => None,
        })
        .collect();
    Ok(geocode.into_series().with_name("geocode".into()))
}

/// Darken a hex color by multiplying its RGB components by `factor`.
///
/// Mirrors `src/colors.py::darken_hex_color`.
#[pyfunction]
#[pyo3(signature = (hex_color, factor = 0.5))]
fn darken_hex_color(hex_color: &str, factor: f64) -> PyResult<String> {
    let stripped = hex_color.trim_start_matches('#');
    let expanded: String = if stripped.len() == 3 {
        stripped.chars().flat_map(|c| [c, c]).collect()
    } else {
        stripped.to_string()
    };
    if expanded.len() < 6 {
        return Err(PyValueError::new_err(format!(
            "invalid hex color: {hex_color:?}"
        )));
    }
    let component = |slice: &str| -> PyResult<u8> {
        u8::from_str_radix(slice, 16)
            .map_err(|e| PyValueError::new_err(format!("invalid hex color {hex_color:?}: {e}")))
    };
    let r = component(&expanded[0..2])?;
    let g = component(&expanded[2..4])?;
    let b = component(&expanded[4..6])?;
    // int(v * factor) truncates toward zero, then clamp to [0, 255] (matches Python).
    let scale = |v: u8| -> u8 { ((v as f64 * factor) as i64).clamp(0, 255) as u8 };
    Ok(format!("#{:02x}{:02x}{:02x}", scale(r), scale(g), scale(b)))
}

/// Given a DataFrame with `decimalLatitude` / `decimalLongitude` f64 columns,
/// return a single-column DataFrame with a `geocode` (UInt64) H3 cell column.
///
/// Mirrors `src/geocode.py::select_geocode_lf`.
#[pyfunction]
fn select_geocode(df: PyDataFrame, precision: u8) -> PyResult<PyDataFrame> {
    let df: DataFrame = df.into();
    let res = resolution_from_u8(precision).map_err(to_py)?;
    let lat = df
        .column("decimalLatitude")
        .map_err(to_py)?
        .as_materialized_series();
    let lng = df
        .column("decimalLongitude")
        .map_err(to_py)?
        .as_materialized_series();
    let geocode = latlng_to_geocode(lat, lng, res).map_err(to_py)?;
    let height = geocode.len();
    let out = DataFrame::new(height, vec![geocode.into_column()]).map_err(to_py)?;
    Ok(PyDataFrame(out))
}

#[pymodule]
fn bioregion_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(darken_hex_color, m)?)?;
    m.add_function(wrap_pyfunction!(select_geocode, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn h3_matches_reference_cell() {
        // Reference value produced by polars_h3.latlng_to_cell(37.77, -122.42, 4).
        let res = Resolution::try_from(4u8).unwrap();
        let cell = u64::from(LatLng::new(37.77, -122.42).unwrap().to_cell(res));
        assert_eq!(cell, 595182179739238399);
    }

    #[test]
    fn darken_halves_components() {
        assert_eq!(darken_hex_color("#ff0000", 0.5).unwrap(), "#7f0000");
        // Shorthand expansion: #f00 -> #ff0000.
        assert_eq!(darken_hex_color("#f00", 0.5).unwrap(), "#7f0000");
    }
}
