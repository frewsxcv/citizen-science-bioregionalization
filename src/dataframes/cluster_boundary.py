import dataframely as dy
import polars as pl
import polars_st
import shapely
from typing import Dict, List

from src.dataframes.geocode_cluster import (
    GeocodeClusterSchema,
    iter_clusters_and_geocodes,
)
from src.dataframes.geocode import GeocodeDataFrame


class ClusterBoundarySchema(dy.Schema):
    cluster = dy.UInt32(nullable=False)
    geometry = dy.Any() # Binary

    @classmethod
    def build(
        cls,
        geocode_cluster_dataframe: dy.DataFrame[GeocodeClusterSchema],
        geocode_dataframe: GeocodeDataFrame,
    ) -> dy.DataFrame["ClusterBoundarySchema"]:
        clusters: List[int] = []
        boundaries: List[shapely.Polygon] = []

        # Create a mapping of geocode to boundary for faster lookup
        geocode_to_boundary: Dict[str, bytes] = {}
        for row in geocode_dataframe.df.select(
            "geocode", "boundary"
        ).iter_rows(named=True):
            geocode_to_boundary[row["geocode"]] = row["boundary"]

        # Iterate through each cluster and combine the boundaries of its geocodes
        for (
            cluster_id,
            geocodes,
        ) in iter_clusters_and_geocodes(geocode_cluster_dataframe):
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
                boundaries.append(cluster_boundary)  # type: ignore

        # Create the dataframe with the correct schema
        df = polars_st.GeoDataFrame(
            data={
                "cluster": pl.Series(clusters).cast(pl.UInt32()),
                "geometry": pl.select(polars_st.from_shapely(pl.Series(boundaries))),
            },
        )

        return ClusterBoundarySchema.validate(df)
