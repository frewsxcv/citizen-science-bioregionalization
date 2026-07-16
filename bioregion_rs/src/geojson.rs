//! Port of `src/geojson.py` (`build_geojson_feature_collection`).
//!
//! Uses the `geojson` crate (its `geo-types` feature converts our existing
//! `geo::Polygon`/`MultiPolygon` directly) rather than hand-building JSON
//! strings: an earlier version of this file did that by hand, but a
//! copy-paste mistake silently swapped the `color`/`fillColor` property
//! values, only caught by an explicit unit test asserting the exact JSON
//! shape. Building typed `Feature`/`FeatureCollection` values and letting the
//! crate serialize them removes that whole class of mistake.
//!
//! `write_geojson` (a plain file write) stays in Python — it's I/O, not
//! compute, and this crate hasn't cut over any pipeline call sites yet (see
//! `RUST_MIGRATION_PLAN.md`). This function returns the FeatureCollection as
//! a JSON string rather than a `geojson.FeatureCollection` Python object: no
//! code in the codebase does an `isinstance` check against that type (it's
//! only ever passed straight through to `geojson.dump`, which just needs
//! something JSON-serializable), so a plain JSON string is a safe stand-in
//! for verifying this function's logic.

use geojson::{Feature, FeatureCollection, Geometry as GeoJsonGeometry, JsonObject, JsonValue};
use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use std::collections::HashMap;

use crate::to_py;
use crate::wkb;

fn geometry_to_geojson(geom: &wkb::Geometry) -> GeoJsonGeometry {
    match geom {
        wkb::Geometry::Polygon(poly) => GeoJsonGeometry::from(poly),
        wkb::Geometry::MultiPolygon(mp) => GeoJsonGeometry::from(mp),
    }
}

/// A single GeoJSON Feature, matching `build_geojson_feature`'s properties
/// and geometry exactly (note: `properties["color"]` is `darkened_color`,
/// and `properties["fillColor"]` is `color` -- that's what Python does).
fn build_feature(cluster: u32, color: &str, darkened_color: &str, geom: &wkb::Geometry) -> Feature {
    let mut properties = JsonObject::new();
    properties.insert("color".to_string(), JsonValue::from(darkened_color));
    properties.insert("fillColor".to_string(), JsonValue::from(color));
    properties.insert("fillOpacity".to_string(), JsonValue::from(0.7));
    properties.insert("weight".to_string(), JsonValue::from(1));
    properties.insert("cluster".to_string(), JsonValue::from(cluster));

    Feature {
        geometry: Some(geometry_to_geojson(geom)),
        properties: Some(properties),
        bbox: None,
        id: None,
        foreign_members: None,
    }
}

/// Build a GeoJSON FeatureCollection (as a JSON string) from cluster
/// boundaries and colors, joined on `cluster` (inner join, matching Python's
/// default `.join()`).
///
/// Mirrors `build_geojson_feature_collection`.
#[pyfunction]
pub fn build_geojson_feature_collection(
    cluster_boundary_df: PyDataFrame,
    cluster_colors_df: PyDataFrame,
) -> PyResult<String> {
    let cluster_boundary_df: DataFrame = cluster_boundary_df.into();
    let cluster_colors_df: DataFrame = cluster_colors_df.into();

    let colors: HashMap<u32, (String, String)> = {
        let cluster_ca = cluster_colors_df
            .column("cluster")
            .map_err(to_py)?
            .as_materialized_series()
            .u32()
            .map_err(to_py)?
            .clone();
        let color_ca = cluster_colors_df
            .column("color")
            .map_err(to_py)?
            .as_materialized_series()
            .str()
            .map_err(to_py)?
            .clone();
        let darkened_ca = cluster_colors_df
            .column("darkened_color")
            .map_err(to_py)?
            .as_materialized_series()
            .str()
            .map_err(to_py)?
            .clone();
        cluster_ca
            .into_no_null_iter()
            .zip(color_ca.iter())
            .zip(darkened_ca.iter())
            .map(|((c, color), darkened)| {
                (
                    c,
                    (
                        color.expect("color column must be non-null").to_string(),
                        darkened
                            .expect("darkened_color column must be non-null")
                            .to_string(),
                    ),
                )
            })
            .collect()
    };

    let cluster_ca = cluster_boundary_df
        .column("cluster")
        .map_err(to_py)?
        .as_materialized_series()
        .u32()
        .map_err(to_py)?
        .clone();
    let geometry_ca = cluster_boundary_df
        .column("geometry")
        .map_err(to_py)?
        .as_materialized_series()
        .binary()
        .map_err(to_py)?
        .clone();

    let mut features: Vec<Feature> = Vec::new();
    for (cluster, geometry_bytes) in cluster_ca.into_no_null_iter().zip(geometry_ca.iter()) {
        let Some((color, darkened_color)) = colors.get(&cluster) else {
            continue; // inner join: skip clusters with no matching color row
        };
        let geom = wkb::decode_geometry(geometry_bytes.expect("geometry column must be non-null"));
        features.push(build_feature(cluster, color, darkened_color, &geom));
    }

    let feature_collection = FeatureCollection {
        bbox: None,
        features,
        foreign_members: None,
    };

    Ok(feature_collection.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_properties_are_not_swapped() {
        // build_geojson_feature maps properties["color"] = darkened_color,
        // properties["fillColor"] = color -- verify we didn't swap them.
        let ring = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)];
        let poly = crate::wkb::decode_polygon(&crate::wkb::polygon(&ring));
        let feature = build_feature(3, "#abcdef", "#556677", &wkb::Geometry::Polygon(poly));
        let properties = feature.properties.unwrap();
        assert_eq!(properties["color"], JsonValue::from("#556677"));
        assert_eq!(properties["fillColor"], JsonValue::from("#abcdef"));
        assert_eq!(properties["cluster"], JsonValue::from(3));
    }

    #[test]
    fn multi_polygon_geometry_round_trips() {
        let ring_a = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)];
        let ring_b = [(5.0, 5.0), (6.0, 5.0), (6.0, 6.0), (5.0, 5.0)];
        let poly_a = crate::wkb::decode_polygon(&crate::wkb::polygon(&ring_a));
        let poly_b = crate::wkb::decode_polygon(&crate::wkb::polygon(&ring_b));
        let mp = geo::MultiPolygon::new(vec![poly_a, poly_b]);
        let bytes = crate::wkb::encode_multi_polygon(&mp);

        let geom = wkb::decode_geometry(&bytes);
        let feature = build_feature(1, "#000000", "#000000", &geom);
        let geojson_geom = feature.geometry.unwrap();
        assert!(matches!(
            geojson_geom.value,
            geojson::GeometryValue::MultiPolygon { .. }
        ));
    }
}
