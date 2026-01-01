from typing import Dict, List

import dataframely as dy
import polars as pl
import polars_st
import shapely

from src.dataframes.geocode import GeocodeNoEdgesSchema
from src.dataframes.geocode_cluster import (
    GeocodeClusterSchema,
    iter_clusters_and_geocodes,
)


class ClusterBoundarySchema(dy.Schema):
    cluster = dy.UInt32(nullable=False)
    geometry = dy.Binary()


def build_cluster_boundary_df(
    geocode_cluster_df: dy.DataFrame[GeocodeClusterSchema],
    geocode_lf: dy.LazyFrame[GeocodeNoEdgesSchema],
) -> dy.DataFrame[ClusterBoundarySchema]:
    """Build cluster boundaries by combining geocode boundaries.

    Creates a single boundary polygon for each cluster by unioning all
    the hexagon boundaries belonging to that cluster.

    Args:
        geocode_cluster_df: DataFrame mapping geocodes to clusters
        geocode_lf: LazyFrame containing geocode boundary geometries

    Returns:
        A validated DataFrame conforming to ClusterBoundarySchema
    """
    clusters: List[int] = []
    boundaries: List[shapely.Polygon] = []

    # Create a mapping of geocode to boundary for faster lookup
    geocode_to_boundary: Dict[str, bytes] = {}
    for row in (
        geocode_lf.select("geocode", "boundary")
        .collect(engine="streaming")
        .iter_rows(named=True)
    ):
        geocode_to_boundary[row["geocode"]] = row["boundary"]

    # Iterate through each cluster and combine the boundaries of its geocodes
    for (
        cluster_id,
        geocodes,
    ) in iter_clusters_and_geocodes(geocode_cluster_df):
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
