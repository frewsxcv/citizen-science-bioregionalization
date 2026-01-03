import dataframely as dy
import polars as pl

from src.dataframes.geocode_neighbors import GeocodeNeighborsSchema


def mock_geocode_neighbors_df() -> dy.DataFrame[GeocodeNeighborsSchema]:
    """
    Creates a mock GeocodeNeighborsSchema DataFrame for testing.

    Creates 5 geocodes with neighbor relationships based on spatial proximity.
    The neighbor map creates a connected graph:
    - 1000 connects to 2000, 3000
    - 2000 connects to 1000, 4000
    - 3000 connects to 1000, 5000
    - 4000 connects to 2000, 5000
    - 5000 connects to 3000, 4000
    """
    # Define neighbor relationships based on spatial proximity
    neighbor_map = {
        1000: [2000, 3000],
        2000: [1000, 4000],
        3000: [1000, 5000],
        4000: [2000, 5000],
        5000: [3000, 4000],
    }

    geocodes = [1000, 2000, 3000, 4000, 5000]

    df = pl.DataFrame(
        {
            "geocode": geocodes,
            "direct_neighbors": [neighbor_map[gc] for gc in geocodes],
            "direct_and_indirect_neighbors": [neighbor_map[gc] for gc in geocodes],
        }
    ).with_columns(
        pl.col("geocode").cast(pl.UInt64),
        pl.col("direct_neighbors").cast(pl.List(pl.UInt64)),
        pl.col("direct_and_indirect_neighbors").cast(pl.List(pl.UInt64)),
    )

    return GeocodeNeighborsSchema.validate(df)
