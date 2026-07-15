//! Minimal little-endian WKB encoders for the geometry types the pipeline emits.
//!
//! Standard ISO WKB (no SRID), which `shapely.from_wkb` and `polars-st` both read.
//! Coordinates are written in (x = longitude, y = latitude) order.

const BYTE_ORDER_LE: u8 = 1;
const TYPE_POINT: u32 = 1;
const TYPE_POLYGON: u32 = 3;

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
}
