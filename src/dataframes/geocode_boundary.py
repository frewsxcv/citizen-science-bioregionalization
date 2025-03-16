import polars as pl
import polars_h3
import polars_st
import shapely
from typing import List, Tuple

from src.data_container import DataContainer, assert_dataframe_schema
from src.dataframes.geocode_cluster import GeocodeClusterDataFrame


class GeocodeBoundaryDataFrame(DataContainer):
    df: polars_st.GeoDataFrame

    SCHEMA = {
        "geocode": pl.Utf8(),
        "boundary": pl.Binary(),
    }

    def __init__(self, df: pl.DataFrame):
        assert_dataframe_schema(df, self.SCHEMA)
        self.df = df

    @classmethod
    def build(
        cls,
        geocode_cluster_dataframe: GeocodeClusterDataFrame,
    ) -> "GeocodeBoundaryDataFrame":
        geocodes: List[str] = []
        polygons: List[shapely.Polygon] = []

        for geocode, boundary in (
            geocode_cluster_dataframe.df.with_columns(
                boundary=polars_h3.cell_to_boundary("geocode")
            )
            .select("geocode", "boundary")
            .iter_rows()
        ):
            polygons.append(
                shapely.Polygon(latlng_list_to_lnglat_list(boundary))
            )
            geocodes.append(geocode)

        df = pl.DataFrame(
            data={
                "geocode": geocodes,
                "boundary": polygons,
            },
        ).with_columns(
            polars_st.from_shapely(pl.col("boundary")).alias("boundary"),
        )
        return cls(df)


def latlng_list_to_lnglat_list(
    latlng_list: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    return [(lng, lat) for lat, lng in latlng_list]

