//! Port of `src/geocode.py`.

use h3o::{LatLng, Resolution};
use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use crate::to_py;

/// Parse an H3 resolution (0-15) from a raw integer.
pub(crate) fn resolution_from_u8(precision: u8) -> PolarsResult<Resolution> {
    Resolution::try_from(precision).map_err(|e| {
        PolarsError::ComputeError(format!("invalid H3 resolution {precision}: {e}").into())
    })
}

/// Map paired lat/lng f64 series to a `geocode` UInt64 series of H3 cell ids.
/// Null in either coordinate, or an invalid coordinate, yields a null geocode.
pub(crate) fn latlng_to_geocode(
    lat: &Series,
    lng: &Series,
    res: Resolution,
) -> PolarsResult<Series> {
    let lat = lat.f64()?;
    let lng = lng.f64()?;
    let geocode: UInt64Chunked = lat
        .iter()
        .zip(lng.iter())
        .map(|(la, lo)| match (la, lo) {
            (Some(la), Some(lo)) => LatLng::new(la, lo).ok().map(|ll| u64::from(ll.to_cell(res))),
            _ => None,
        })
        .collect();
    Ok(geocode.into_series().with_name("geocode".into()))
}

/// Return the f64 lat/lng columns of a DataFrame by name.
pub(crate) fn latlng_columns<'a>(
    df: &'a DataFrame,
    lat_col: &str,
    lng_col: &str,
) -> PolarsResult<(&'a Series, &'a Series)> {
    let lat = df.column(lat_col)?.as_materialized_series();
    let lng = df.column(lng_col)?.as_materialized_series();
    Ok((lat, lng))
}

/// Given a DataFrame with `decimalLatitude` / `decimalLongitude` f64 columns,
/// return a single-column DataFrame with a `geocode` (UInt64) H3 cell column.
///
/// Mirrors `src/geocode.py::select_geocode_lf`.
#[pyfunction]
pub fn select_geocode(df: PyDataFrame, precision: u8) -> PyResult<PyDataFrame> {
    let df: DataFrame = df.into();
    let res = resolution_from_u8(precision).map_err(to_py)?;
    let (lat, lng) = latlng_columns(&df, "decimalLatitude", "decimalLongitude").map_err(to_py)?;
    let geocode = latlng_to_geocode(lat, lng, res).map_err(to_py)?;
    let height = geocode.len();
    let out = DataFrame::new(height, vec![geocode.into_column()]).map_err(to_py)?;
    Ok(PyDataFrame(out))
}

/// Return the input DataFrame with an added `geocode` (UInt64) column.
///
/// Mirrors `src/geocode.py::with_geocode_lf`.
#[pyfunction]
pub fn with_geocode(df: PyDataFrame, precision: u8) -> PyResult<PyDataFrame> {
    let mut df: DataFrame = df.into();
    let res = resolution_from_u8(precision).map_err(to_py)?;
    let (lat, lng) = latlng_columns(&df, "decimalLatitude", "decimalLongitude").map_err(to_py)?;
    let geocode = latlng_to_geocode(lat, lng, res).map_err(to_py)?;
    df.with_column(geocode.into_column()).map_err(to_py)?;
    Ok(PyDataFrame(df))
}

/// Filter a DataFrame to rows whose lat/lng are non-null and within the bounding
/// box (inclusive on both bounds).
///
/// Mirrors `src/geocode.py::filter_by_bounding_box`.
#[pyfunction]
#[pyo3(signature = (
    df, min_lat, max_lat, min_lng, max_lng,
    lat_col = "decimalLatitude".to_string(),
    lng_col = "decimalLongitude".to_string(),
))]
#[allow(clippy::too_many_arguments)]
pub fn filter_by_bounding_box(
    df: PyDataFrame,
    min_lat: f64,
    max_lat: f64,
    min_lng: f64,
    max_lng: f64,
    lat_col: String,
    lng_col: String,
) -> PyResult<PyDataFrame> {
    let df: DataFrame = df.into();
    let (lat, lng) = latlng_columns(&df, &lat_col, &lng_col).map_err(to_py)?;
    let lat = lat.f64().map_err(to_py)?;
    let lng = lng.f64().map_err(to_py)?;
    let mask: BooleanChunked = lat
        .iter()
        .zip(lng.iter())
        .map(|(la, lo)| match (la, lo) {
            (Some(la), Some(lo)) => {
                la >= min_lat && la <= max_lat && lo >= min_lng && lo <= max_lng
            }
            _ => false,
        })
        .collect();
    let out = df.filter(&mask).map_err(to_py)?;
    Ok(PyDataFrame(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn h3_matches_reference_cell() {
        // Reference value produced by polars_h3.latlng_to_cell(37.77, -122.42, 4).
        let res = resolution_from_u8(4).unwrap();
        let cell = u64::from(LatLng::new(37.77, -122.42).unwrap().to_cell(res));
        assert_eq!(cell, 595182179739238399);
    }
}
