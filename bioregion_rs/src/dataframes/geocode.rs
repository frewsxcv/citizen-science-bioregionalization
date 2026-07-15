//! Port of `src/dataframes/geocode.py` (the `build_geocode_*` geometry builder).

use geo::{Intersects, LineString, Polygon};
use h3o::{CellIndex, LatLng};
use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use crate::geocode::{latlng_columns, latlng_to_geocode, resolution_from_u8};
use crate::{to_py, wkb};

/// Closed boundary ring of an H3 cell as (lng, lat) coordinates.
fn cell_ring(cell: CellIndex) -> Vec<(f64, f64)> {
    let boundary = cell.boundary();
    let mut ring: Vec<(f64, f64)> = boundary.iter().map(|v| (v.lng(), v.lat())).collect();
    if let Some(&first) = ring.first() {
        ring.push(first); // close the ring
    }
    ring
}

/// Build a GeocodeSchema DataFrame from occurrence coordinates.
///
/// Mirrors `src/dataframes/geocode.py::build_geocode_lf`: geocode the input
/// coordinates, keep the unique non-null cells sorted ascending, and for each
/// emit its center point, boundary polygon (both WKB), and whether it is an
/// "edge" cell (its boundary intersects the bounding-box boundary).
///
/// Output columns match GeocodeSchema order: geocode, center, boundary, is_edge.
#[pyfunction]
#[pyo3(signature = (df, precision, min_lat, max_lat, min_lng, max_lng))]
pub fn build_geocode(
    df: PyDataFrame,
    precision: u8,
    min_lat: f64,
    max_lat: f64,
    min_lng: f64,
    max_lng: f64,
) -> PyResult<PyDataFrame> {
    let df: DataFrame = df.into();
    let res = resolution_from_u8(precision).map_err(to_py)?;
    let (lat, lng) = latlng_columns(&df, "decimalLatitude", "decimalLongitude").map_err(to_py)?;
    let geocodes = latlng_to_geocode(lat, lng, res).map_err(to_py)?;

    // Unique, non-null, sorted ascending — matches `.filter(is_not_null).unique().sort()`.
    let mut cells: Vec<u64> = geocodes.u64().map_err(to_py)?.iter().flatten().collect();
    cells.sort_unstable();
    cells.dedup();

    // Bounding-box boundary as a closed linestring (lng, lat), same 5 corners as Python.
    let bbox_ring = LineString::from(vec![
        (min_lng, min_lat),
        (max_lng, min_lat),
        (max_lng, max_lat),
        (min_lng, max_lat),
        (min_lng, min_lat),
    ]);

    let mut centers: Vec<Vec<u8>> = Vec::with_capacity(cells.len());
    let mut boundaries: Vec<Vec<u8>> = Vec::with_capacity(cells.len());
    let mut is_edge: Vec<bool> = Vec::with_capacity(cells.len());

    for &raw in &cells {
        let cell = CellIndex::try_from(raw).map_err(|e| {
            to_py(PolarsError::ComputeError(
                format!("invalid H3 cell {raw}: {e}").into(),
            ))
        })?;

        let center = LatLng::from(cell);
        centers.push(wkb::point(center.lng(), center.lat()));

        let ring = cell_ring(cell);
        boundaries.push(wkb::polygon(&ring));

        let polygon = Polygon::new(LineString::from(ring), vec![]);
        is_edge.push(polygon.intersects(&bbox_ring));
    }

    let geocode_col = UInt64Chunked::from_vec("geocode".into(), cells).into_column();
    let center_col = binary_column("center", &centers);
    let boundary_col = binary_column("boundary", &boundaries);
    let is_edge_col = BooleanChunked::from_iter_values("is_edge".into(), is_edge.into_iter())
        .into_column();

    let height = geocode_col.len();
    let out = DataFrame::new(
        height,
        vec![geocode_col, center_col, boundary_col, is_edge_col],
    )
    .map_err(to_py)?;
    Ok(PyDataFrame(out))
}

/// Build a named Binary column from a slice of byte vectors.
fn binary_column(name: &str, values: &[Vec<u8>]) -> Column {
    let ca: BinaryChunked = values.iter().map(|v| Some(v.as_slice())).collect();
    ca.into_series().with_name(name.into()).into_column()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_is_closed() {
        let res = resolution_from_u8(8).unwrap();
        let cell = LatLng::new(45.0, 7.5).unwrap().to_cell(res);
        let ring = cell_ring(cell);
        assert!(ring.len() >= 7); // hexagon: 6 vertices + closing point
        assert_eq!(ring.first(), ring.last());
    }
}
