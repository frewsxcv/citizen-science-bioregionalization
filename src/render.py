from typing import Any, Dict, List

import polars as pl
import shapely


def features_to_polars_df(features: List[Dict[str, Any]]) -> pl.DataFrame:
    """
    Convert GeoJSON features to a polars DataFrame with geometry column.

    Args:
        features: List of GeoJSON features

    Returns:
        Polars DataFrame with geometry and properties columns
    """
    rows = []
    for feature in features:
        # Convert GeoJSON geometry to shapely geometry and then to WKB binary format
        shapely_geom = shapely.geometry.shape(feature["geometry"])
        row = {"geometry": shapely.to_wkb(shapely_geom)}

        # Add all the properties
        row.update(feature["properties"])
        rows.append(row)

    return pl.DataFrame(rows)


def darken_hex_color(hex_color: str, factor: float = 0.5) -> str:
    """
    Darken a hex color by a factor.

    Args:
        hex_color: Hex color string like "#ff0000" or "#f00"
        factor: Factor to darken by (0.0 to 1.0, where 0.0 is black, 1.0 is unchanged)

    Returns:
        Darkened hex color string
    """
    # Normalize shorthand form
    if len(hex_color) == 4:
        hex_color = f"#{hex_color[1]}{hex_color[1]}{hex_color[2]}{hex_color[2]}{hex_color[3]}{hex_color[3]}"

    # Extract RGB components
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)

    # Apply darkening factor
    r = int(r * factor)
    g = int(g * factor)
    b = int(b * factor)

    # Ensure values are in valid range
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))

    # Convert back to hex
    return f"#{r:02x}{g:02x}{b:02x}"


def darken_hex_colors_polars(colors: pl.Series, factor: float = 0.5) -> pl.Series:
    """
    Darken all hex colors in a polars Series.

    Args:
        colors: Polars Series containing hex color strings
        factor: Factor to darken by (0.0 to 1.0, where 0.0 is black, 1.0 is unchanged)

    Returns:
        Polars Series with darkened hex colors
    """
    return colors.map_elements(
        lambda color: darken_hex_color(color, factor), return_dtype=pl.Utf8
    )
