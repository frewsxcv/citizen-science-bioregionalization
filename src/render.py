import geopandas
import matplotlib.pyplot as plt
import contextily
import geojson


def plot_clusters(feature_collection: geojson.FeatureCollection) -> None:
    geojson_gdf = geopandas.GeoDataFrame.from_features(
        feature_collection["features"], crs="EPSG:4326"
    )
    geojson_gdf_wm = geojson_gdf.to_crs(epsg=3857)

    # Note: This does not yet honor the fill color in the geojson
    ax = geojson_gdf_wm.plot(
        column=geojson_gdf_wm["cluster"],
        legend=True,
        categorical=True,
        alpha=0.5,
    )
    contextily.add_basemap(
        ax, source=contextily.providers.CartoDB.Positron, attribution_size=0
    )
    ax.set_axis_off()
    plt.show()
