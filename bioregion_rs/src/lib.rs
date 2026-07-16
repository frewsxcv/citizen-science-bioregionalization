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

mod cluster_optimization;
mod colors;
mod dataframes;
mod geocode;
mod geojson;
mod matrices;
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
    m.add_function(wrap_pyfunction!(
        matrices::geocode_connectivity::build_geocode_connectivity_matrix,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        dataframes::cluster_taxa_statistics::build_cluster_taxa_statistics,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        dataframes::cluster_neighbors::build_cluster_neighbors,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        matrices::cluster_distance::build_cluster_distance_matrix,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        dataframes::cluster_color::build_cluster_color,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        dataframes::cluster_boundary::build_cluster_boundary,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        dataframes::cluster_significant_differences::build_cluster_significant_differences,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        dataframes::permanova_results::build_permanova_results,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        dataframes::geocode_cluster_metrics::build_geocode_cluster_metrics,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        dataframes::geocode_cluster_metrics::select_optimal_k_elbow,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        dataframes::geocode_silhouette_score::build_geocode_silhouette_score,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        cluster_optimization::optimize_num_clusters,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        geojson::build_geojson_feature_collection,
        m
    )?)?;
    Ok(())
}
