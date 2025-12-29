import dataframely as dy
import polars as pl
import polars_st as pl_st
import shapely

from src.dataframes.geocode import GeocodeNoEdgesSchema


def mock_geocode_no_edges_df() -> dy.DataFrame[GeocodeNoEdgesSchema]:
    """
    Creates a mock GeocodeNoEdgesSchema DataFrame for testing.

    Creates 5 geocodes in a simple spatial arrangement with connectivity.
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

    # Define neighbor relationships based on spatial proximity
    # 0 connects to 1, 2
    # 1 connects to 0, 3
    # 2 connects to 0, 4
    # 3 connects to 1, 4
    # 4 connects to 2, 3
    neighbor_map = {
        1000: [2000, 3000],
        2000: [1000, 4000],
        3000: [1000, 5000],
        4000: [2000, 5000],
        5000: [3000, 4000],
    }

    # Create DataFrame with base columns first
    df = pl.DataFrame(
        {
            "geocode": [1000, 2000, 3000, 4000, 5000],
            "direct_neighbors": [
                neighbor_map[gc] for gc in [1000, 2000, 3000, 4000, 5000]
            ],
            "direct_and_indirect_neighbors": [
                neighbor_map[gc] for gc in [1000, 2000, 3000, 4000, 5000]
            ],
            "is_edge": [False, False, False, False, False],
        }
    ).with_columns(
        pl.col("geocode").cast(pl.UInt64),
        pl.col("direct_neighbors").cast(pl.List(pl.UInt64)),
        pl.col("direct_and_indirect_neighbors").cast(pl.List(pl.UInt64)),
        pl.col("is_edge").cast(pl.Boolean),
    )

    # Add geometry columns using expressions
    df = df.with_columns(
        pl_st.from_shapely(pl.Series(centers)).alias("center"),
        pl_st.from_shapely(pl.Series(boundaries)).alias("boundary"),
    )

    return GeocodeNoEdgesSchema.validate(df)
