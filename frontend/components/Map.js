import React, { useRef, useEffect } from "react";

const data = require("../aggregations.json");

const Map = ({ setSelectedCluster }) => {
  const mapContainer = useRef(null);
  const map = useRef(null);

  useEffect(() => {
    if (map.current) return; // initialize map only once
    map.current = new maplibregl.Map({
      container: mapContainer.current,
      style:
        "https://api.maptiler.com/maps/streets/style.json?key=get_your_own_OpIi9ZULNHzrESv6T2vL",
      center: [-20, 65],
      zoom: 4,
    });

    map.current.on("load", () => {
      const geojson = {
        type: "FeatureCollection",
        features: data.map((cluster) => ({
          type: "Feature",
          geometry: cluster.boundary,
          properties: {
            cluster: cluster.cluster,
            significant_taxa: cluster.significant_taxa,
            color: cluster.color,
            darkened_color: cluster.darkened_color,
          },
        })),
      };

      map.current.addSource("clusters", {
        type: "geojson",
        data: geojson,
      });

      map.current.addLayer({
        id: "clusters-fill",
        type: "fill",
        source: "clusters",
        paint: {
          "fill-color": ["get", "color"],
          "fill-opacity": 0.7,
        },
      });

      map.current.addLayer({
        id: "clusters-outline",
        type: "line",
        source: "clusters",
        paint: {
          "line-color": ["get", "darkened_color"],
          "line-width": 2,
        },
      });

      map.current.on("click", "clusters-fill", (e) => {
        const properties = e.features[0].properties;
        const significantTaxa = JSON.parse(properties.significant_taxa);
        setSelectedCluster({
          clusterId: properties.cluster,
          significantTaxa: significantTaxa,
        });
      });

      map.current.on("mouseenter", "clusters-fill", () => {
        map.current.getCanvas().style.cursor = "pointer";
      });

      map.current.on("mouseleave", "clusters-fill", () => {
        map.current.getCanvas().style.cursor = "";
      });
    });
  });

  return <div ref={mapContainer} id="map" />;
};

export default Map;
