//! Port of `src/dataframes/taxonomy.py`.

use std::collections::HashSet;

use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use crate::geocode::{filter_by_bounding_box_df, with_geocode_df};
use crate::to_py;

/// Set of known geocodes as a plain u64 hash set, for semi-join-style filtering.
pub(crate) fn known_geocodes(geocode_df: &DataFrame) -> PolarsResult<HashSet<u64>> {
    Ok(geocode_df
        .column("geocode")?
        .as_materialized_series()
        .u64()?
        .into_no_null_iter()
        .collect())
}

/// Build a mask selecting rows of `df` whose `geocode` column is in `known`.
fn known_geocode_mask(df: &DataFrame, known: &HashSet<u64>) -> PolarsResult<BooleanChunked> {
    let geocode = df.column("geocode")?.as_materialized_series().u64()?;
    Ok(geocode
        .iter()
        .map(|g| g.is_some_and(|g| known.contains(&g)))
        .collect())
}

/// Build a TaxonomySchema DataFrame from Darwin Core occurrence data.
///
/// Mirrors `src/dataframes/taxonomy.py::build_taxonomy_lf`: distinct
/// (scientificName, gbifTaxonId) pairs among occurrences that fall within a
/// known (non-edge) geocode, each assigned a synthetic `taxonId`.
///
/// Note: row order (and therefore the `taxonId` assigned to each pair) is not
/// guaranteed to match the Python implementation's `.unique()` ordering — only
/// the set of pairs and the bijection with `taxonId` are guaranteed.
#[pyfunction]
#[pyo3(signature = (darwin_core_df, geocode_precision, geocode_df, min_lat, max_lat, min_lng, max_lng))]
#[allow(clippy::too_many_arguments)]
pub fn build_taxonomy(
    darwin_core_df: PyDataFrame,
    geocode_precision: u8,
    geocode_df: PyDataFrame,
    min_lat: f64,
    max_lat: f64,
    min_lng: f64,
    max_lng: f64,
) -> PyResult<PyDataFrame> {
    let darwin_core_df: DataFrame = darwin_core_df.into();
    let geocode_df: DataFrame = geocode_df.into();

    let mut df = darwin_core_df
        .select([
            "decimalLatitude",
            "decimalLongitude",
            "scientificName",
            "taxonKey",
        ])
        .map_err(to_py)?;
    df.rename("taxonKey", "gbifTaxonId".into()).map_err(to_py)?;

    let df = filter_by_bounding_box_df(
        &df,
        min_lat,
        max_lat,
        min_lng,
        max_lng,
        "decimalLatitude",
        "decimalLongitude",
    )
    .map_err(to_py)?;
    let df = with_geocode_df(&df, geocode_precision).map_err(to_py)?;

    let known = known_geocodes(&geocode_df).map_err(to_py)?;
    let mask = known_geocode_mask(&df, &known).map_err(to_py)?;
    let df = df.filter(&mask).map_err(to_py)?;

    let df = df
        .select(["scientificName", "gbifTaxonId"])
        .map_err(to_py)?;
    let df = df
        .unique::<&str, String>(None, UniqueKeepStrategy::Any, None)
        .map_err(to_py)?;
    let df = df.with_row_index("taxonId".into(), None).map_err(to_py)?;

    Ok(PyDataFrame(df))
}
