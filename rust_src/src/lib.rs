use pyo3::prelude::*;
use pyo3::exceptions::PyValueError; // Import standard Python exceptions
use pyo3_polars::{PyDataFrame, PyLazyFrame};
use polars::prelude::*;

/// Helper function to convert PolarsError to PyErr
fn polars_err_to_py_err(e: PolarsError) -> PyErr {
    PyValueError::new_err(format!("Polars error: {}", e))
}

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// Builds the taxonomy dataframe using Rust Polars operations.
#[pyfunction]
fn build_taxonomy_dataframe_rust(lazy_frame: PyLazyFrame) -> PyResult<PyDataFrame> {
    let lf: LazyFrame = lazy_frame.into();

    let df = lf.select([
            col("kingdom"),
            col("phylum"),
            col("class"),
            col("order"),
            col("family"),
            col("genus"),
            col("species"),
            col("taxonRank"),
            col("scientificName"),
        ])
        .unique(None, UniqueKeepStrategy::First) // Keep the first unique row
        .collect() // Collect into a DataFrame
        .map_err(polars_err_to_py_err)? // Map PolarsError to PyErr
        .lazy() // Convert back to LazyFrame to use with_row_index
        .with_row_index("taxonId", Some(0)) // Add taxonId starting from 0
        .with_column(col("taxonId").cast(DataType::UInt32)) // Cast taxonId to UInt32
        .collect() // Collect final DataFrame
        .map_err(polars_err_to_py_err)?; // Map PolarsError to PyErr

    Ok(PyDataFrame(df))
}


/// A Python module implemented in Rust.
#[pymodule]
fn rust_dataframe_utils(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(build_taxonomy_dataframe_rust, m)?)?; // Add the new function
    Ok(())
}
