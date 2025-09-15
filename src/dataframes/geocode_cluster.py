from typing import Iterator, List, Tuple
import logging
import polars as pl
import dataframely as dy
from scipy.sparse import csr_matrix
from sklearn.cluster import AgglomerativeClustering
from src.matrices.geocode_connectivity import GeocodeConnectivityMatrix
from src.dataframes.geocode import GeocodeDataFrame
from src.types import Geocode, ClusterId
from src.matrices.geocode_distance import GeocodeDistanceMatrix

logger = logging.getLogger(__name__)


class GeocodeClusterSchema(dy.Schema):
    geocode = dy.UInt64(nullable=False)
    cluster = dy.UInt32(nullable=False)

    @classmethod
    def build(
        cls,
        geocode_dataframe: GeocodeDataFrame,
        distance_matrix: GeocodeDistanceMatrix,
        connectivity_matrix: GeocodeConnectivityMatrix,
        num_clusters: int,
    ) -> dy.DataFrame["GeocodeClusterSchema"]:
        geocodes = geocode_dataframe.df["geocode"]
        clusters = AgglomerativeClustering(
            n_clusters=num_clusters,
            connectivity=csr_matrix(connectivity_matrix._connectivity_matrix),  # type: ignore
            linkage="ward",
        ).fit_predict(distance_matrix.squareform())
        assert len(geocodes) == len(clusters)
        df = pl.DataFrame(
            data={
                "geocode": geocodes,
                "cluster": clusters,
            }
        ).with_columns(pl.col("cluster").cast(pl.UInt32))
        return cls.validate(df)


def cluster_ids(
    geocode_cluster_dataframe: dy.DataFrame[GeocodeClusterSchema],
) -> List[ClusterId]:
    return geocode_cluster_dataframe["cluster"].unique().to_list()


def iter_clusters_and_geocodes(
    geocode_cluster_dataframe: dy.DataFrame[GeocodeClusterSchema],
) -> Iterator[Tuple[ClusterId, List[str]]]:
    for row in (
        geocode_cluster_dataframe.group_by("cluster").all().sort("cluster")
    ).iter_rows(named=True):
        yield row["cluster"], row["geocode"]


def cluster_for_geocode(
    geocode_cluster_dataframe: dy.DataFrame[GeocodeClusterSchema], geocode: Geocode
) -> ClusterId:
    return geocode_cluster_dataframe.filter(pl.col("geocode") == geocode)[
        "cluster"
    ].to_list()[0]


def geocodes_for_cluster(
    geocode_cluster_dataframe: dy.DataFrame[GeocodeClusterSchema], cluster: ClusterId
) -> List[Geocode]:
    return geocode_cluster_dataframe.filter(pl.col("cluster") == cluster)[
        "geocode"
    ].to_list()


def num_clusters(
    geocode_cluster_dataframe: dy.DataFrame[GeocodeClusterSchema],
) -> int:
    num = geocode_cluster_dataframe["cluster"].max()
    assert isinstance(num, int)
    return num
