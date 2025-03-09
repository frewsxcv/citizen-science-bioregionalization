import polars as pl
from src.data_container import DataContainer
from src.dataframes.cluster_neighbors import ClusterNeighborsDataFrame
from sklearn.metrics import silhouette_score, silhouette_samples

from src.dataframes.geocode_cluster import GeocodeClusterDataFrame
from src.matrices.distance import DistanceMatrix


class GeocodeSilhouetteScoreDataFrame(DataContainer):
    df: pl.DataFrame

    SCHEMA = {
        "geocode": pl.String(),
        "silhouette_score": pl.Float64(),
    }

    def __init__(self, df: pl.DataFrame) -> None:
        assert df.schema == self.SCHEMA
        self.df = df

    @classmethod
    def build(
        cls,
        cluster_neighbors_dataframe: ClusterNeighborsDataFrame,
        distance_matrix: DistanceMatrix,
        geocode_cluster_dataframe: GeocodeClusterDataFrame,
    ) -> "GeocodeSilhouetteScoreDataFrame":
        geocodes: list[str | None] = []
        silhouette_scores: list[float] = []

        # The first entry will be for all geocodes, with cluster=null
        geocodes.append(None)
        silhouette_scores.append(silhouette_score(
            X=distance_matrix.squareform(),
            labels=geocode_cluster_dataframe.df["cluster"],
            metric="precomputed",
        ))

        # Add the clusters and their scores
        geocodes.extend(geocode_cluster_dataframe.df["geocode"])
        silhouette_scores.extend(silhouette_samples(
            X=distance_matrix.squareform(),
            labels=geocode_cluster_dataframe.df["cluster"],
            metric="precomputed",
        ))

        df = pl.DataFrame({
            "geocode": geocodes,
            "silhouette_score": silhouette_scores,
        }, schema=cls.SCHEMA)

        return cls(df)