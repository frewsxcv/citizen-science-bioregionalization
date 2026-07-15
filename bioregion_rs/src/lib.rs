//! `bioregion_rs` — Rust core for the bioregionalization pipeline.
//!
//! Modules mirror the Python source layout (see `RUST_MIGRATION_PLAN.md`):
//! - `colors`             <- `src/colors.py`
//! - `geocode`            <- `src/geocode.py`
//! - `dataframes::geocode` <- `src/dataframes/geocode.py`
//! - `wkb`                 -- WKB encoders for geometry columns

use polars::prelude::PolarsError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

mod colors;
mod dataframes;
mod geocode;
mod wkb;

/// Convert a polars error into a Python `ValueError`.
pub(crate) fn to_py(e: PolarsError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

#[pymodule]
fn bioregion_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(colors::darken_hex_color, m)?)?;
    m.add_function(wrap_pyfunction!(geocode::select_geocode, m)?)?;
    m.add_function(wrap_pyfunction!(geocode::with_geocode, m)?)?;
    m.add_function(wrap_pyfunction!(geocode::filter_by_bounding_box, m)?)?;
    m.add_function(wrap_pyfunction!(dataframes::geocode::build_geocode, m)?)?;
    m.add_function(wrap_pyfunction!(dataframes::taxonomy::build_taxonomy, m)?)?;
    m.add_function(wrap_pyfunction!(
        dataframes::geocode_taxa_counts::build_geocode_taxa_counts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        dataframes::geocode_neighbors::build_geocode_neighbors,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        dataframes::geocode_neighbors::build_geocode_neighbors_no_edges,
        m
    )?)?;
    Ok(())
}
