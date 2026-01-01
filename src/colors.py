"""Color utility functions for the citizen science bioregionalization project.

This module provides functions for manipulating hex color values,
including darkening colors for visual distinction in maps and plots.
"""

import polars as pl


def darken_hex_color(hex_color: str, factor: float = 0.5) -> str:
    """
    Darken a hex color by multiplying RGB components by the given factor.

    Args:
        hex_color: A hex color string like '#ff0000' or '#f00'
        factor: A float between 0 and 1 (0 = black, 1 = original color)

    Returns:
        A darkened hex color string
    """
    # Remove the # if present
    hex_color = hex_color.lstrip("#")

    # Handle shorthand hex format (#rgb -> #rrggbb)
    if len(hex_color) == 3:
        hex_color = "".join([c * 2 for c in hex_color])

    # Convert hex to RGB
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    # Apply darkening factor and ensure values are in valid range
    r = max(0, min(255, int(r * factor)))
    g = max(0, min(255, int(g * factor)))
    b = max(0, min(255, int(b * factor)))

    # Convert back to hex
    return f"#{r:02x}{g:02x}{b:02x}"


def darken_hex_colors_polars(colors: pl.Series, factor: float = 0.5) -> pl.Series:
    """
    Darken all hex colors in a Polars Series.

    Args:
        colors: Polars Series containing hex color strings
        factor: Factor to darken by (0.0 to 1.0, where 0.0 is black, 1.0 is unchanged)

    Returns:
        Polars Series with darkened hex colors
    """
    return colors.map_elements(
        lambda color: darken_hex_color(color, factor), return_dtype=pl.Utf8
    )
