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
