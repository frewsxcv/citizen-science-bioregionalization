import React, { useState } from "react";
import {
  useClusterData,
  useSelectedClusterId,
  useSelectClusterById,
} from "../store/selectors";
import { flyToCluster, exportClusterData } from "../store/actions";

/**
 * ClusterNavigator - Example component demonstrating advanced Zustand features
 *
 * This component shows:
 * - Using selectors for state access
 * - Using actions for complex operations
 * - Combining multiple state slices
 * - Conditional rendering based on state
 */
const ClusterNavigator: React.FC = () => {
  const clusterData = useClusterData();
  const selectedClusterId = useSelectedClusterId();
  const selectClusterById = useSelectClusterById();
  const [showNavigator, setShowNavigator] = useState(false);

  if (!showNavigator) {
    return (
      <button
        onClick={() => setShowNavigator(true)}
        style={{
          position: "absolute",
          top: "60px",
          right: "10px",
          zIndex: 1000,
          padding: "8px 12px",
          background: "white",
          border: "1px solid #ccc",
          borderRadius: "4px",
          cursor: "pointer",
          fontSize: "12px",
        }}
        title="Open cluster navigator"
      >
        🗺️ Navigate
      </button>
    );
  }

  return (
    <div
      style={{
        position: "absolute",
        top: "60px",
        right: "10px",
        zIndex: 1000,
        background: "white",
        border: "1px solid #ccc",
        borderRadius: "8px",
        padding: "12px",
        maxWidth: "300px",
        maxHeight: "400px",
        overflow: "auto",
        boxShadow: "0 2px 8px rgba(0,0,0,0.1)",
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: "12px",
        }}
      >
        <h4 style={{ margin: 0, fontSize: "14px", fontWeight: "600" }}>
          Cluster Navigator
        </h4>
        <button
          onClick={() => setShowNavigator(false)}
          style={{
            background: "none",
            border: "none",
            fontSize: "16px",
            cursor: "pointer",
            padding: "0 4px",
          }}
          title="Close"
        >
          ✕
        </button>
      </div>

      <div style={{ fontSize: "12px", color: "#666", marginBottom: "12px" }}>
        {clusterData.length} clusters available
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
        {clusterData.map((cluster) => {
          const isSelected = cluster.cluster === selectedClusterId;
          const taxaCount = cluster.significant_taxa.length;

          return (
            <div
              key={cluster.cluster}
              style={{
                display: "flex",
                alignItems: "center",
                gap: "8px",
                padding: "8px",
                background: isSelected ? "#e3f2fd" : "#f9f9f9",
                border: isSelected ? "2px solid #2196f3" : "1px solid #e0e0e0",
                borderRadius: "4px",
                cursor: "pointer",
                transition: "all 0.2s",
              }}
              onClick={() => selectClusterById(cluster.cluster)}
            >
              <div
                style={{
                  width: "24px",
                  height: "24px",
                  borderRadius: "4px",
                  background: cluster.color,
                  border: `2px solid ${cluster.darkened_color}`,
                  flexShrink: 0,
                }}
              />

              <div style={{ flex: 1, fontSize: "12px" }}>
                <div style={{ fontWeight: isSelected ? "600" : "500" }}>
                  Cluster {cluster.cluster}
                </div>
                <div style={{ fontSize: "11px", color: "#666" }}>
                  {taxaCount} taxa
                </div>
              </div>

              <div style={{ display: "flex", gap: "4px" }}>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    flyToCluster(cluster.cluster);
                  }}
                  style={{
                    padding: "4px 6px",
                    fontSize: "10px",
                    background: "#fff",
                    border: "1px solid #ccc",
                    borderRadius: "3px",
                    cursor: "pointer",
                  }}
                  title="Fly to cluster on map"
                >
                  🎯
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    exportClusterData(cluster.cluster);
                  }}
                  style={{
                    padding: "4px 6px",
                    fontSize: "10px",
                    background: "#fff",
                    border: "1px solid #ccc",
                    borderRadius: "3px",
                    cursor: "pointer",
                  }}
                  title="Export cluster data"
                >
                  💾
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default ClusterNavigator;
