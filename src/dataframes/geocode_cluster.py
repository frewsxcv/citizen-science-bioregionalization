from typing import Iterator, List, Tuple
import logging
import polars as pl
from scipy.sparse import csr_matrix
from sklearn.cluster import AgglomerativeClustering
from src.matrices.connectivity import ConnectivityMatrix
from src.dataframes.geocode import GeocodeDataFrame
from src.types import Geocode, ClusterId
from src.matrices.distance import DistanceMatrix
from src.data_container import DataContainer

logger = logging.getLogger(__name__)


class GeocodeClusterDataFrame(DataContainer):
    df: pl.DataFrame

    SCHEMA = {
        "geocode": pl.String,
        "cluster": pl.UInt32,
    }

    def __init__(self, df: pl.DataFrame) -> None:
        self.df = df

    @classmethod
    def build(
        cls,
        geocode_dataframe: GeocodeDataFrame,
        distance_matrix: DistanceMatrix,
        connectivity_matrix: ConnectivityMatrix,
        num_clusters: int,
    ) -> 'GeocodeClusterDataFrame':
        geocodes = geocode_dataframe.df["geocode"]
        clusters = AgglomerativeClustering(
            n_clusters=num_clusters,
            connectivity=csr_matrix(connectivity_matrix._connectivity_matrix),
            linkage="ward",
        ).fit_predict(distance_matrix.squareform())
        assert len(geocodes) == len(clusters)
        return cls(
            df=pl.DataFrame(
                data={
                    "geocode": geocodes,
                    "cluster": clusters,
                },
                schema=cls.SCHEMA,
            )
        )

    def cluster_ids(self) -> List[ClusterId]:
        return self.df["cluster"].unique().to_list()

    def iter_clusters_and_geocodes(
        self,
    ) -> Iterator[Tuple[ClusterId, List[str]]]:
        for row in (self.df.group_by("cluster").all().sort("cluster")).iter_rows(
            named=True
        ):
            yield row["cluster"], row["geocode"]

    def cluster_for_geocode(self, geocode: Geocode) -> ClusterId:
        return self.df.filter(pl.col("geocode") == geocode)["cluster"].to_list()[0]

    def geocodes_for_cluster(self, cluster: ClusterId) -> List[Geocode]:
        return self.df.filter(pl.col("cluster") == cluster)["geocode"].to_list()

    def num_clusters(self) -> int:
        num = self.df["cluster"].max()
        assert isinstance(num, int)
        return num
