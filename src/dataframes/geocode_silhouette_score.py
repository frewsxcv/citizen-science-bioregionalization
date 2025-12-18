import dataframely as dy
import polars as pl
from sklearn.metrics import silhouette_samples, silhouette_score  # type: ignore

from src.dataframes.geocode_cluster import GeocodeClusterSchema
from src.matrices.geocode_distance import GeocodeDistanceMatrix


class GeocodeSilhouetteScoreSchema(dy.Schema):
    geocode = dy.UInt64(nullable=True)
    silhouette_score = dy.Float64(nullable=False)

    @classmethod
    def build(
        cls,
        distance_matrix: GeocodeDistanceMatrix,
        geocode_cluster_dataframe: dy.DataFrame[GeocodeClusterSchema],
    ) -> dy.DataFrame["GeocodeSilhouetteScoreSchema"]:
        geocodes: list[int | None] = []
        silhouette_scores: list[float] = []

        # The first entry will be for all geocodes, with cluster=null
        geocodes.append(None)
        silhouette_scores.append(
            float(
                silhouette_score(
                    X=distance_matrix.squareform(),
                    labels=geocode_cluster_dataframe["cluster"],
                    metric="precomputed",
                )
            )
        )

        # Add the clusters and their scores
        geocodes.extend(geocode_cluster_dataframe["geocode"])
        samples = silhouette_samples(
            X=distance_matrix.squareform(),
            labels=geocode_cluster_dataframe["cluster"],
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
