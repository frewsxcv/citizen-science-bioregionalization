import React, { useRef, useEffect } from "react";
import type { FeatureCollection } from "geojson";
import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { ClusterData } from "../types";
import dataImport from "../aggregations.json";
import { imageLoadQueue } from "../utils/ImageLoadQueue";
import { useStore } from "../store/useStore";

const data = dataImport as ClusterData[];

const Map: React.FC = () => {
  const mapContainer = useRef<HTMLDivElement>(null);
  const map = useRef<maplibregl.Map | null>(null);

  const setSelectedCluster = useStore((state) => state.setSelectedCluster);
  const setMapInstance = useStore((state) => state.setMapInstance);
  const setClusterData = useStore((state) => state.setClusterData);

  useEffect(() => {
    // Load cluster data into the store
    setClusterData(data);
  }, []); // setClusterData is stable from Zustand

  useEffect(() => {
    if (map.current || !mapContainer.current) return; // initialize map only once

    map.current = new maplibregl.Map({
      container: mapContainer.current,
      style:
        "https://api.maptiler.com/maps/streets/style.json?key=get_your_own_OpIi9ZULNHzrESv6T2vL",
      center: [-20, 65],
      zoom: 4,
    });

    map.current.on("load", () => {
      if (!map.current) return;

      // Store the map instance in the global store
      setMapInstance(map.current);

      const geojson: FeatureCollection = {
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

      map.current.on("click", "clusters-fill", (e: any) => {
        const properties = e.features[0].properties;
        const significantTaxa = JSON.parse(properties.significant_taxa);

        // Clear the image queue to prioritize images from the newly selected cluster
        imageLoadQueue.clear();

        setSelectedCluster({
          clusterId: properties.cluster,
          significantTaxa: significantTaxa,
        });
      });

      map.current.on("mouseenter", "clusters-fill", () => {
        if (map.current) {
          map.current.getCanvas().style.cursor = "pointer";
        }
      });

      map.current.on("mouseleave", "clusters-fill", () => {
        if (map.current) {
          map.current.getCanvas().style.cursor = "";
        }
      });
    });

    // Cleanup on unmount
    return () => {
      if (map.current) {
        map.current.remove();
        setMapInstance(null);
      }
    };
  }, []); // Zustand actions are stable and don't need to be in dependencies

  return <div ref={mapContainer} id="map" />;
};

export default Map;
