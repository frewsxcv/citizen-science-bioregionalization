import polars as pl
import geojson  # type: ignore
from typing import Optional, List, Dict, Any, Self
import dataframely as dy

from src.data_container import DataContainer, assert_dataframe_schema
from src.types import ClusterId
from src.dataframes.cluster_boundary import ClusterBoundarySchema
from src.dataframes.cluster_color import ClusterColorDataFrame


class ClusterGeojsonFeaturesDataFrame(DataContainer):
    df: pl.DataFrame

    SCHEMA = {
        "cluster": pl.UInt32(),
        "feature": pl.Utf8(),  # Serialized GeoJSON Feature
    }

    def __init__(self, df: pl.DataFrame) -> None:
        assert_dataframe_schema(df, self.SCHEMA)
        self.df = df

    def get_feature_for_cluster(
        self, cluster_id: ClusterId
    ) -> Optional[geojson.Feature]:
        """Get the GeoJSON feature for a specific cluster."""
        # Filter for the specific cluster
        filtered_df = self.df.filter(pl.col("cluster") == cluster_id)

        if filtered_df.height == 0:
            return None

        feature_str = filtered_df.select("feature").item()
        return geojson.loads(feature_str)

    @classmethod
    def build(
        cls,
        cluster_boundary_dataframe: dy.DataFrame[ClusterBoundarySchema],
        cluster_colors_dataframe: ClusterColorDataFrame,
    ) -> Self:
        """
        Build a ClusterGeojsonFeaturesDataFrame from cluster boundaries and colors.

        Args:
            cluster_boundary_dataframe: Dataframe containing cluster boundary data
            cluster_colors_dataframe: Dataframe containing cluster color data

        Returns:
            A ClusterGeojsonFeaturesDataFrame with clusters and their GeoJSON features
        """
        import shapely

        clusters = []
        features = []

        # Join the two dataframes to get all information needed for each cluster
        joined_df = cluster_boundary_dataframe.join(
            cluster_colors_dataframe.df, on="cluster"
        )

        # Process each cluster to create its GeoJSON feature
        for row in joined_df.iter_rows(named=True):
            cluster_id = row["cluster"]
            geometry = shapely.from_wkb(row["geometry"])
            color = row["color"]
            darkened_color = row["darkened_color"]

            # Build the GeoJSON feature
            feature = geojson.Feature(
                properties={
                    "color": darkened_color,
                    "fillColor": color,
                    "fillOpacity": 0.7,
                    "weight": 1,
                    "cluster": int(cluster_id),
                },
                geometry=shapely.geometry.mapping(geometry),  # type: ignore
            )

            clusters.append(cluster_id)
            features.append(geojson.dumps(feature))

        # Create the dataframe with the correct schema
        df = pl.DataFrame(
            {
                "cluster": pl.Series(clusters, dtype=pl.UInt32),
                "feature": pl.Series(features, dtype=pl.Utf8),
            }
        )

        return cls(df)

    def to_feature_collection(self) -> geojson.FeatureCollection:
        """Convert all features in the dataframe to a GeoJSON FeatureCollection."""
        features = []

        for feature_str in self.df["feature"]:
            feature = geojson.loads(feature_str)
            features.append(feature)

        return geojson.FeatureCollection(features)
