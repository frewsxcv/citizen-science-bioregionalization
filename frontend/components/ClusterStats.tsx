import React from "react";
import {
  useClusterStats,
  useHasSelection,
  useSelectedClusterId,
} from "../store/selectors";

const ClusterStats: React.FC = () => {
  const hasSelection = useHasSelection();
  const clusterId = useSelectedClusterId();
  const stats = useClusterStats();

  if (!hasSelection || !stats) {
    return null;
  }

  return (
    <div
      style={{
        background: "#f5f5f5",
        padding: "12px",
        borderRadius: "8px",
        marginBottom: "16px",
        fontSize: "14px",
      }}
    >
      <h4 style={{ margin: "0 0 12px 0", fontSize: "16px" }}>
        Cluster {clusterId} Statistics
      </h4>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: "8px",
        }}
      >
        <StatItem
          label="Total Taxa"
          value={stats.totalTaxaCount.toString()}
        />
        <StatItem
          label="Total Observations"
          value={stats.totalObservations.toLocaleString()}
        />
        <StatItem
          label="With Images"
          value={`${stats.taxaWithImages} (${Math.round((stats.taxaWithImages / stats.totalTaxaCount) * 100)}%)`}
        />
        <StatItem
          label="Without Images"
          value={stats.taxaWithoutImages.toString()}
        />
        <StatItem
          label="Avg P-Value"
          value={stats.averagePValue.toExponential(2)}
        />
        <StatItem
          label="Avg Log2 FC"
          value={stats.averageFoldChange.toFixed(2)}
        />
      </div>
    </div>
  );
};

interface StatItemProps {
  label: string;
  value: string;
}

const StatItem: React.FC<StatItemProps> = ({ label, value }) => {
  return (
    <div>
      <div
        style={{
          fontSize: "11px",
          color: "#666",
          textTransform: "uppercase",
          marginBottom: "2px",
        }}
      >
        {label}
      </div>
      <div style={{ fontWeight: "600", color: "#333" }}>{value}</div>
    </div>
  );
};

export default ClusterStats;
