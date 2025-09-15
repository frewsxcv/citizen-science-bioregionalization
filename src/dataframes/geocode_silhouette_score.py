import polars as pl
import dataframely as dy
import numpy as np
from src.dataframes.cluster_neighbors import ClusterNeighborsSchema
from sklearn.metrics import silhouette_score, silhouette_samples  # type: ignore

from src.dataframes.geocode_cluster import GeocodeClusterDataFrame
from src.matrices.geocode_distance import GeocodeDistanceMatrix


class GeocodeSilhouetteScoreSchema(dy.Schema):
    geocode = dy.UInt64(nullable=True)
    silhouette_score = dy.Float64(nullable=False)

    @classmethod
    def build(
        cls,
        cluster_neighbors_dataframe: dy.DataFrame[ClusterNeighborsSchema],
        distance_matrix: GeocodeDistanceMatrix,
        geocode_cluster_dataframe: GeocodeClusterDataFrame,
    ) -> dy.DataFrame["GeocodeSilhouetteScoreSchema"]:
        geocodes: list[int | None] = []
        silhouette_scores: list[float] = []

        # The first entry will be for all geocodes, with cluster=null
        geocodes.append(None)
        silhouette_scores.append(
            float(
                silhouette_score(
                    X=distance_matrix.squareform(),
                    labels=geocode_cluster_dataframe.df["cluster"],
                    metric="precomputed",
                )
            )
        )

        # Add the clusters and their scores
        geocodes.extend(geocode_cluster_dataframe.df["geocode"])
        samples = silhouette_samples(
            X=distance_matrix.squareform(),
            labels=geocode_cluster_dataframe.df["cluster"],
            metric="precomputed",
        )
        silhouette_scores.extend(list(samples))  # type: ignore

        df = pl.DataFrame(
            {
                "geocode": geocodes,
                "silhouette_score": silhouette_scores,
            }
        ).with_columns(pl.col("geocode").cast(pl.UInt64, strict=False))

        return cls.validate(df)
