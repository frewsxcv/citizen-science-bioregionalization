import dataframely as dy
import polars as pl
from sklearn.metrics import silhouette_samples, silhouette_score  # type: ignore

from src.dataframes.geocode_cluster import GeocodeClusterMultiKSchema
from src.matrices.geocode_distance import GeocodeDistanceMatrix


class GeocodeSilhouetteScoreSchema(dy.Schema):
    geocode = dy.UInt64(nullable=True)
    silhouette_score = dy.Float64(nullable=False)
    num_clusters = dy.UInt32(nullable=False)

    @classmethod
    def build_df(
        cls,
        distance_matrix: GeocodeDistanceMatrix,
        geocode_cluster_df: dy.DataFrame[GeocodeClusterMultiKSchema],
    ) -> dy.DataFrame["GeocodeSilhouetteScoreSchema"]:
        """Build silhouette scores for all clustering results.

        Args:
            distance_matrix: Precomputed distance matrix between geocodes
            geocode_cluster_df: DataFrame with clustering results for all k values

        Returns:
            DataFrame with silhouette scores for all k values tested
        """
        all_results = []

        # Get unique k values from the geocode_cluster_df
        k_values = geocode_cluster_df["num_clusters"].unique().sort()

        for k in k_values:
            # Filter to just this k value
            k_df = geocode_cluster_df.filter(pl.col("num_clusters") == k)

            geocodes: list[int | None] = []
            silhouette_scores: list[float] = []

            # The first entry will be for all geocodes, with cluster=null
            geocodes.append(None)
            silhouette_scores.append(
                float(
                    silhouette_score(
                        X=distance_matrix.squareform(),
                        labels=k_df["cluster"],
                        metric="precomputed",
                    )
                )
            )

            # Add the clusters and their scores
            geocodes.extend(k_df["geocode"])
            samples = silhouette_samples(
                X=distance_matrix.squareform(),
                labels=k_df["cluster"],
                metric="precomputed",
            )
            silhouette_scores.extend(list(samples))  # type: ignore

            k_result = pl.DataFrame(
                {
                    "geocode": geocodes,
                    "silhouette_score": silhouette_scores,
                    "num_clusters": [k] * len(geocodes),
                }
            ).with_columns(
                pl.col("geocode").cast(pl.UInt64, strict=False),
                pl.col("num_clusters").cast(pl.UInt32),
            )

            all_results.append(k_result)

        df = pl.concat(all_results)
        return cls.validate(df)
