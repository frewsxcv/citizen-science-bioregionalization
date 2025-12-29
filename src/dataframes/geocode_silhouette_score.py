import dataframely as dy
import polars as pl
from sklearn.metrics import silhouette_samples, silhouette_score  # type: ignore

from src.dataframes.geocode_cluster import GeocodeClusterSchema
from src.matrices.geocode_distance import GeocodeDistanceMatrix


class GeocodeSilhouetteScoreSchema(dy.Schema):
    geocode = dy.UInt64(nullable=True)
    silhouette_score = dy.Float64(nullable=False)
    num_clusters = dy.UInt32(nullable=False)

    @classmethod
    def build_df(
        cls,
        distance_matrix: GeocodeDistanceMatrix,
        geocode_cluster_df: dy.DataFrame[GeocodeClusterSchema],
        num_clusters: int,
    ) -> dy.DataFrame["GeocodeSilhouetteScoreSchema"]:
        geocodes: list[int | None] = []
        silhouette_scores: list[float] = []

        # The first entry will be for all geocodes, with cluster=null
        geocodes.append(None)
        silhouette_scores.append(
            float(
                silhouette_score(
                    X=distance_matrix.squareform(),
                    labels=geocode_cluster_df["cluster"],
                    metric="precomputed",
                )
            )
        )

        # Add the clusters and their scores
        geocodes.extend(geocode_cluster_df["geocode"])
        samples = silhouette_samples(
            X=distance_matrix.squareform(),
            labels=geocode_cluster_df["cluster"],
            metric="precomputed",
        )
        silhouette_scores.extend(list(samples))  # type: ignore

        df = pl.DataFrame(
            {
                "geocode": geocodes,
                "silhouette_score": silhouette_scores,
                "num_clusters": [num_clusters] * len(geocodes),
            }
        ).with_columns(
            pl.col("geocode").cast(pl.UInt64, strict=False),
            pl.col("num_clusters").cast(pl.UInt32),
        )

        return cls.validate(df)
