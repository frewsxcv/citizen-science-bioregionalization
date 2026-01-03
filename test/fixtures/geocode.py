import dataframely as dy
import polars as pl
import polars_st as pl_st
import shapely

from src.dataframes.geocode import GeocodeNoEdgesSchema


def mock_geocode_no_edges_df() -> dy.DataFrame[GeocodeNoEdgesSchema]:
    """
    Creates a mock GeocodeNoEdgesSchema DataFrame for testing.

    Creates 5 geocodes in a simple spatial arrangement.
    """
    # Create simple point geometries
    centers = [
        shapely.Point(-70.0, 42.0),
        shapely.Point(-70.1, 42.0),
        shapely.Point(-70.0, 42.1),
        shapely.Point(-70.1, 42.1),
        shapely.Point(-70.05, 42.05),
    ]

    # Create simple hexagonal boundaries (approximated as small polygons)
    boundaries = []
    for center in centers:
        x, y = center.x, center.y
        hex_points = [
            (x + 0.01, y),
            (x + 0.005, y + 0.01),
            (x - 0.005, y + 0.01),
            (x - 0.01, y),
            (x - 0.005, y - 0.01),
            (x + 0.005, y - 0.01),
        ]
        boundaries.append(shapely.Polygon(hex_points))

    # Create DataFrame with base columns first
    df = pl.DataFrame(
        {
            "geocode": [1000, 2000, 3000, 4000, 5000],
            "is_edge": [False, False, False, False, False],
        }
    ).with_columns(
        pl.col("geocode").cast(pl.UInt64),
        pl.col("is_edge").cast(pl.Boolean),
    )

    # Add geometry columns using expressions
    df = df.with_columns(
        pl_st.from_shapely(pl.Series(centers)).alias("center"),
        pl_st.from_shapely(pl.Series(boundaries)).alias("boundary"),
    )

    return GeocodeNoEdgesSchema.validate(df)
