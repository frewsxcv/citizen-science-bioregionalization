import polars as pl
import polars_st
import shapely
from typing import Dict, List, Tuple

from src.data_container import DataContainer, assert_dataframe_schema
from src.dataframes.geocode_cluster import GeocodeClusterDataFrame
from src.dataframes.geocode_boundary import GeocodeBoundaryDataFrame
from src.types import ClusterId


class ClusterBoundaryDataFrame(DataContainer):
    df: polars_st.GeoDataFrame

    SCHEMA = {
        "cluster": pl.UInt32(),
        "boundary": pl.Binary(),
    }

    def __init__(self, df: pl.DataFrame):
        assert_dataframe_schema(df, self.SCHEMA)
        self.df = df

    @classmethod
    def build(
        cls,
        geocode_cluster_dataframe: GeocodeClusterDataFrame,
        geocode_boundary_dataframe: GeocodeBoundaryDataFrame,
    ) -> "ClusterBoundaryDataFrame":
        clusters: List[int] = []
        boundaries: List[shapely.Polygon] = []

        # Create a mapping of geocode to boundary for faster lookup
        geocode_to_boundary: Dict[str, bytes] = {}
        for row in geocode_boundary_dataframe.df.select(
            "geocode", "boundary"
        ).iter_rows(named=True):
            geocode_to_boundary[row["geocode"]] = row["boundary"]

        # Iterate through each cluster and combine the boundaries of its geocodes
        for (
            cluster_id,
            geocodes,
        ) in geocode_cluster_dataframe.iter_clusters_and_geocodes():
            # Get all geocode boundaries for this cluster
            cluster_geocode_boundaries = []
            for geocode in geocodes:
                if geocode in geocode_to_boundary:
                    # Convert the binary WKB to a shapely geometry
                    geom = shapely.from_wkb(geocode_to_boundary[geocode])
                    if geom is not None:
                        cluster_geocode_boundaries.append(geom)

            if cluster_geocode_boundaries:
                # Union all polygons to create a single boundary for the cluster
                if len(cluster_geocode_boundaries) == 1:
                    cluster_boundary = cluster_geocode_boundaries[0]
                else:
                    # First dissolve/union all geometries
                    cluster_boundary = shapely.unary_union(cluster_geocode_boundaries)

                clusters.append(cluster_id)
                boundaries.append(cluster_boundary)

        # Create the dataframe with the correct schema
        df = pl.DataFrame(
            data={
                "cluster": clusters,
                "boundary": boundaries,
            },
        )

        # Cast to the correct types to match our schema
        df = df.with_columns(
            [
                pl.col("cluster").cast(pl.UInt32()),
                polars_st.from_shapely(pl.col("boundary")).alias("boundary"),
            ]
        )

        return cls(df)

    def get_boundary_for_cluster(self, cluster_id: ClusterId) -> shapely.Polygon:
        """Get the boundary polygon for a specific cluster."""
        boundary = (
            self.df.filter(pl.col("cluster") == cluster_id).select("boundary").item()
        )
        return boundary
