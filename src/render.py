import geopandas
import matplotlib.pyplot as plt
import contextily
import geojson
from matplotlib.lines import Line2D


def plot_clusters(
    feature_collection: geojson.FeatureCollection, num_clusters: int
) -> None:
    geojson_gdf = geopandas.GeoDataFrame.from_features(
        feature_collection["features"], crs="EPSG:4326"
    )
    geojson_gdf_wm = geojson_gdf.to_crs(epsg=3857)

    ax = geojson_gdf_wm.plot(
        color=geojson_gdf_wm["fill"],
        categorical=True,
        alpha=0.5,
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
