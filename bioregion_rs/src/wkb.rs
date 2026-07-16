//! Minimal little-endian WKB encoders/decoders for the geometry types the
//! pipeline emits.
//!
//! Standard ISO WKB (no SRID), which `shapely.from_wkb` and `polars-st` both read.
//! Coordinates are written in (x = longitude, y = latitude) order.

use geo::{Coord, LineString, MultiPolygon, Polygon};

const BYTE_ORDER_LE: u8 = 1;
const TYPE_POINT: u32 = 1;
const TYPE_POLYGON: u32 = 3;
const TYPE_MULTIPOLYGON: u32 = 6;

/// Encode a 2D point as WKB.
pub fn point(x: f64, y: f64) -> Vec<u8> {
    let mut buf = Vec::with_capacity(21);
    buf.push(BYTE_ORDER_LE);
    buf.extend_from_slice(&TYPE_POINT.to_le_bytes());
    buf.extend_from_slice(&x.to_le_bytes());
    buf.extend_from_slice(&y.to_le_bytes());
    buf
}

/// Encode a single-exterior-ring polygon as WKB. `ring` must already be closed
/// (first coordinate equal to last) and ordered (x = lng, y = lat).
pub fn polygon(ring: &[(f64, f64)]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(1 + 4 + 4 + 4 + ring.len() * 16);
    buf.push(BYTE_ORDER_LE);
    buf.extend_from_slice(&TYPE_POLYGON.to_le_bytes());
    buf.extend_from_slice(&1u32.to_le_bytes()); // one ring
    buf.extend_from_slice(&(ring.len() as u32).to_le_bytes());
    for (x, y) in ring {
        buf.extend_from_slice(&x.to_le_bytes());
        buf.extend_from_slice(&y.to_le_bytes());
    }
    buf
}

/// Decode a WKB-encoded 2D point back to (x, y). Panics on malformed input;
/// only intended for round-tripping this module's own `point()` output.
pub fn decode_point(bytes: &[u8]) -> (f64, f64) {
    assert_eq!(
        bytes[0], BYTE_ORDER_LE,
        "only little-endian WKB is supported"
    );
    let kind = u32::from_le_bytes(bytes[1..5].try_into().unwrap());
    assert_eq!(kind, TYPE_POINT, "expected a WKB Point");
    let x = f64::from_le_bytes(bytes[5..13].try_into().unwrap());
    let y = f64::from_le_bytes(bytes[13..21].try_into().unwrap());
    (x, y)
}

/// Decode a WKB Polygon (any number of rings: exterior + holes) into a
/// `geo::Polygon`. Panics on malformed input or an unexpected geometry type.
pub fn decode_polygon(bytes: &[u8]) -> Polygon<f64> {
    decode_polygon_at(bytes, 0).0
}

/// Decode a WKB Polygon starting at byte offset `start`, returning the
/// polygon and the offset immediately after it (needed to decode further
/// geometries packed into the same buffer, e.g. a MultiPolygon's elements).
fn decode_polygon_at(bytes: &[u8], start: usize) -> (Polygon<f64>, usize) {
    assert_eq!(
        bytes[start], BYTE_ORDER_LE,
        "only little-endian WKB is supported"
    );
    let kind = u32::from_le_bytes(bytes[start + 1..start + 5].try_into().unwrap());
    assert_eq!(kind, TYPE_POLYGON, "expected a WKB Polygon");
    let num_rings = u32::from_le_bytes(bytes[start + 5..start + 9].try_into().unwrap()) as usize;
    let mut offset = start + 9;
    let mut rings: Vec<LineString<f64>> = Vec::with_capacity(num_rings);
    for _ in 0..num_rings {
        let num_points = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        let mut coords: Vec<Coord<f64>> = Vec::with_capacity(num_points);
        for _ in 0..num_points {
            let x = f64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
            let y = f64::from_le_bytes(bytes[offset + 8..offset + 16].try_into().unwrap());
            coords.push(Coord { x, y });
            offset += 16;
        }
        rings.push(LineString::new(coords));
    }
    let exterior = rings.remove(0);
    (Polygon::new(exterior, rings), offset)
}

/// Decode a WKB MultiPolygon (each element is itself a full WKB Polygon
/// buffer, per the WKB MultiPolygon spec) into a `geo::MultiPolygon`.
pub fn decode_multi_polygon(bytes: &[u8]) -> MultiPolygon<f64> {
    assert_eq!(
        bytes[0], BYTE_ORDER_LE,
        "only little-endian WKB is supported"
    );
    let kind = u32::from_le_bytes(bytes[1..5].try_into().unwrap());
    assert_eq!(kind, TYPE_MULTIPOLYGON, "expected a WKB MultiPolygon");
    let num_polygons = u32::from_le_bytes(bytes[5..9].try_into().unwrap()) as usize;
    let mut offset = 9;
    let mut polygons = Vec::with_capacity(num_polygons);
    for _ in 0..num_polygons {
        let (poly, next_offset) = decode_polygon_at(bytes, offset);
        polygons.push(poly);
        offset = next_offset;
    }
    MultiPolygon::new(polygons)
}

/// Either shape a `cluster_boundary`-style WKB blob can take: a bare Polygon
/// (single-geocode clusters) or a MultiPolygon (`geo::unary_union`'s return
/// type for multi-geocode clusters, see `dataframes/cluster_boundary.rs`).
pub enum Geometry {
    Polygon(Polygon<f64>),
    MultiPolygon(MultiPolygon<f64>),
}

/// Decode a WKB Polygon or MultiPolygon, dispatching on its type tag. Panics
/// on malformed input or an unsupported geometry type.
pub fn decode_geometry(bytes: &[u8]) -> Geometry {
    assert_eq!(
        bytes[0], BYTE_ORDER_LE,
        "only little-endian WKB is supported"
    );
    let kind = u32::from_le_bytes(bytes[1..5].try_into().unwrap());
    match kind {
        TYPE_POLYGON => Geometry::Polygon(decode_polygon(bytes)),
        TYPE_MULTIPOLYGON => Geometry::MultiPolygon(decode_multi_polygon(bytes)),
        other => panic!("unsupported WKB geometry type {other}"),
    }
}

/// Encode a `geo::Polygon` (exterior + any holes) as WKB.
pub fn encode_polygon(poly: &Polygon<f64>) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.push(BYTE_ORDER_LE);
    buf.extend_from_slice(&TYPE_POLYGON.to_le_bytes());
    let interiors = poly.interiors();
    buf.extend_from_slice(&(1 + interiors.len() as u32).to_le_bytes());
    encode_ring(&mut buf, poly.exterior());
    for ring in interiors {
        encode_ring(&mut buf, ring);
    }
    buf
}

fn encode_ring(buf: &mut Vec<u8>, ring: &LineString<f64>) {
    buf.extend_from_slice(&(ring.0.len() as u32).to_le_bytes());
    for coord in &ring.0 {
        buf.extend_from_slice(&coord.x.to_le_bytes());
        buf.extend_from_slice(&coord.y.to_le_bytes());
    }
}

/// Encode a `geo::MultiPolygon` as WKB (each element is itself a full WKB
/// Polygon buffer, per the WKB MultiPolygon spec).
pub fn encode_multi_polygon(mp: &MultiPolygon<f64>) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.push(BYTE_ORDER_LE);
    buf.extend_from_slice(&TYPE_MULTIPOLYGON.to_le_bytes());
    buf.extend_from_slice(&(mp.0.len() as u32).to_le_bytes());
    for poly in &mp.0 {
        buf.extend_from_slice(&encode_polygon(poly));
    }
    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_wkb_layout() {
        let b = point(1.0, 2.0);
        assert_eq!(b.len(), 21);
        assert_eq!(b[0], BYTE_ORDER_LE);
        assert_eq!(u32::from_le_bytes(b[1..5].try_into().unwrap()), TYPE_POINT);
        assert_eq!(f64::from_le_bytes(b[5..13].try_into().unwrap()), 1.0);
        assert_eq!(f64::from_le_bytes(b[13..21].try_into().unwrap()), 2.0);
        assert_eq!(decode_point(&b), (1.0, 2.0));
    }

    #[test]
    fn polygon_wkb_layout() {
        let ring = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)];
        let b = polygon(&ring);
        assert_eq!(b[0], BYTE_ORDER_LE);
        assert_eq!(
            u32::from_le_bytes(b[1..5].try_into().unwrap()),
            TYPE_POLYGON
        );
        assert_eq!(u32::from_le_bytes(b[5..9].try_into().unwrap()), 1);
        assert_eq!(u32::from_le_bytes(b[9..13].try_into().unwrap()), 4);
    }

    #[test]
    fn decode_polygon_round_trips_through_encode() {
        let ring = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)];
        let decoded = decode_polygon(&polygon(&ring));
        assert_eq!(decoded.interiors().len(), 0);
        let coords: Vec<(f64, f64)> = decoded.exterior().coords().map(|c| (c.x, c.y)).collect();
        assert_eq!(coords, ring);

        let re_encoded = encode_polygon(&decoded);
        assert_eq!(decode_polygon(&re_encoded).exterior(), decoded.exterior());
    }

    #[test]
    fn decode_multi_polygon_round_trips_through_encode() {
        let ring_a = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)];
        let ring_b = [(5.0, 5.0), (6.0, 5.0), (6.0, 6.0), (5.0, 5.0)];
        let poly_a = decode_polygon(&polygon(&ring_a));
        let poly_b = decode_polygon(&polygon(&ring_b));
        let mp = MultiPolygon::new(vec![poly_a, poly_b]);

        let encoded = encode_multi_polygon(&mp);
        let decoded = decode_multi_polygon(&encoded);
        assert_eq!(decoded.0.len(), 2);
        assert_eq!(decoded.0[0].exterior(), mp.0[0].exterior());
        assert_eq!(decoded.0[1].exterior(), mp.0[1].exterior());

        match decode_geometry(&encoded) {
            Geometry::MultiPolygon(_) => {}
            Geometry::Polygon(_) => panic!("expected MultiPolygon"),
        }
        match decode_geometry(&polygon(&ring_a)) {
            Geometry::Polygon(_) => {}
            Geometry::MultiPolygon(_) => panic!("expected Polygon"),
        }
    }
}
