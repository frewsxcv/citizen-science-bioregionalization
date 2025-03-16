import geopandas
import matplotlib.pyplot as plt
import contextily
import geojson
from matplotlib.lines import Line2D
import io
import base64
import os


def plot_clusters(
    feature_collection: geojson.FeatureCollection,
) -> None:
    geojson_gdf = geopandas.GeoDataFrame.from_features(
        feature_collection["features"], crs="EPSG:4326"
    )
    geojson_gdf_wm = geojson_gdf.to_crs(epsg=3857)

    ax = geojson_gdf_wm.plot(
        color=geojson_gdf_wm["fill"],
        categorical=True,
        linewidth=0,
        alpha=0.5,
    )
    geojson_gdf_wm.boundary.plot(
        ax=ax,
        color=geojson_gdf_wm["fill"],
        linewidth=0.5,
        alpha=1,
    )

    # Add a legend
    cluster_and_fill = geojson_gdf_wm[["cluster", "fill"]]
    custom_points = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=10, color=row.fill)
        for row in cluster_and_fill.itertuples()
    ]
    leg_points = ax.legend(custom_points, cluster_and_fill["cluster"].unique())
    ax.add_artist(leg_points)

    contextily.add_basemap(
        ax, source=contextily.providers.CartoDB.Positron, attribution_size=0
    )
    ax.set_axis_off()
    plt.show()
    # os.makedirs("output", exist_ok=True)
    # plt.savefig(f"output/{num_clusters}.png")


def plot_single_cluster(
    feature_collection: geojson.FeatureCollection,
    cluster_id: int,
    save_path: str | None = None,
    to_base64: bool = False,
) -> str:
    """
    Plot a single cluster and optionally save it to a file or return as base64 encoded image.
    
    Args:
        feature_collection: GeoJSON feature collection containing all clusters
        cluster_id: The ID of the cluster to plot
        save_path: Path to save the image to (optional)
        to_base64: Whether to return the image as a base64 encoded string
        
    Returns:
        Base64 encoded image string if to_base64 is True, otherwise empty string
    """
    # Filter features for the specific cluster
    cluster_features = [
        feature for feature in feature_collection["features"]
        if feature["properties"]["cluster"] == cluster_id
    ]
    
    # Create a new feature collection with just this cluster
    cluster_feature_collection = geojson.FeatureCollection(cluster_features)
    
    # Convert to GeoDataFrame
    geojson_gdf = geopandas.GeoDataFrame.from_features(
        cluster_features, crs="EPSG:4326"
    )
    geojson_gdf_wm = geojson_gdf.to_crs(epsg=3857)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot the cluster
    geojson_gdf_wm.plot(
        ax=ax,
        color=geojson_gdf_wm["fill"],
        categorical=True,
        linewidth=0,
        alpha=0.5,
    )
    geojson_gdf_wm.boundary.plot(
        ax=ax,
        color=geojson_gdf_wm["fill"],
        linewidth=0.5,
        alpha=1,
    )
    
    # Add basemap
    contextily.add_basemap(
        ax, source=contextily.providers.CartoDB.Positron, attribution_size=0
    )
    
    # Set title and remove axes
    ax.set_title(f"Cluster {cluster_id}")
    ax.set_axis_off()
    
    # Tight layout
    plt.tight_layout()
    
    # Handle output
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
    
    if to_base64:
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('ascii')
        plt.close(fig)
        return img_str
    
    plt.close(fig)
    return ""
