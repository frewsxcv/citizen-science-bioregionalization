//! Port of `src/colors.py`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Darken a hex color by multiplying its RGB components by `factor`.
///
/// Mirrors `src/colors.py::darken_hex_color`.
#[pyfunction]
#[pyo3(signature = (hex_color, factor = 0.5))]
pub fn darken_hex_color(hex_color: &str, factor: f64) -> PyResult<String> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn darken_halves_components() {
        assert_eq!(darken_hex_color("#ff0000", 0.5).unwrap(), "#7f0000");
        // Shorthand expansion: #f00 -> #ff0000.
        assert_eq!(darken_hex_color("#f00", 0.5).unwrap(), "#7f0000");
    }
}
