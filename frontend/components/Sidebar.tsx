import React, { useState, useMemo } from "react";
import ImageWithRetry from "./ImageWithRetry";
import ClusterStats from "./ClusterStats";
import {
  useSelectedCluster,
  useSidebarOpen,
  useClearSelection,
} from "../store/selectors";

interface TaxaItemProps {
  taxa: {
    gbif_taxon_id: number;
    taxon_id: number;
    scientific_name: string;
    image_url: string | null;
    p_value: number;
    log2_fold_change: number;
    cluster_count: number;
    neighbor_count: number;
  };
}

interface TaxaListProps {
  significantTaxa: TaxaItemProps["taxa"][];
  sortBy: "count" | "pvalue" | "foldchange";
}

interface ClusterDetailsProps {
  clusterId: number;
}

const TaxaItem: React.FC<TaxaItemProps> = ({ taxa }) => {
  return (
    <li key={taxa.gbif_taxon_id}>
      {taxa.image_url ? (
        <ImageWithRetry
          src={taxa.image_url}
          alt={taxa.scientific_name}
          style={{
            width: "50px",
            height: "50px",
            objectFit: "cover",
            marginRight: "10px",
            float: "left",
          }}
        />
      ) : (
        <div
          style={{
            width: "50px",
            height: "50px",
            float: "left",
            marginRight: "10px",
            background: "#eee",
          }}
        ></div>
      )}
      {taxa.scientific_name}
      <br />
      <small>
        p={taxa.p_value.toExponential(2)}, log2fc=
        {taxa.log2_fold_change.toFixed(2)}, in-cluster=
        {taxa.cluster_count}, neighbors={taxa.neighbor_count}
      </small>
      <div style={{ clear: "both" }}></div>
    </li>
  );
};

const TaxaList: React.FC<TaxaListProps> = ({ significantTaxa, sortBy }) => {
  // Sort inside component using useMemo to avoid infinite loops
  const sortedTaxa = useMemo(() => {
    if (significantTaxa.length === 0) return [];

    const taxa = [...significantTaxa];

    switch (sortBy) {
      case "count":
        return taxa.sort((a, b) => b.cluster_count - a.cluster_count);
      case "pvalue":
        return taxa.sort((a, b) => a.p_value - b.p_value);
      case "foldchange":
        return taxa.sort(
          (a, b) => Math.abs(b.log2_fold_change) - Math.abs(a.log2_fold_change),
        );
      default:
        return taxa;
    }
  }, [significantTaxa, sortBy]);

  if (significantTaxa.length === 0) {
    return <p>No significant taxa for this cluster.</p>;
  }

  return (
    <ul>
      {sortedTaxa.map((taxa) => (
        <TaxaItem key={taxa.taxon_id} taxa={taxa} />
      ))}
    </ul>
  );
};

const ClusterDetails: React.FC<ClusterDetailsProps> = ({ clusterId }) => {
  const clearSelection = useClearSelection();
  const selectedCluster = useSelectedCluster();
  const [sortBy, setSortBy] = useState<"count" | "pvalue" | "foldchange">(
    "count",
  );

  if (!selectedCluster) return null;

  return (
    <>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: "12px",
        }}
      >
        <h3 style={{ margin: 0 }}>Cluster {clusterId}</h3>
        <button
          onClick={clearSelection}
          style={{
            padding: "4px 8px",
            background: "#f44336",
            color: "white",
            border: "none",
            borderRadius: "4px",
            cursor: "pointer",
            fontSize: "12px",
          }}
        >
          Clear
        </button>
      </div>

      <ClusterStats />

      <div style={{ marginBottom: "12px" }}>
        <label
          htmlFor="sort-select"
          style={{ fontSize: "12px", color: "#666", marginRight: "8px" }}
        >
          Sort by:
        </label>
        <select
          id="sort-select"
          value={sortBy}
          onChange={(e) =>
            setSortBy(e.target.value as "count" | "pvalue" | "foldchange")
          }
          style={{
            padding: "4px 8px",
            borderRadius: "4px",
            border: "1px solid #ccc",
            fontSize: "12px",
          }}
        >
          <option value="count">Observation Count</option>
          <option value="pvalue">P-Value</option>
          <option value="foldchange">Fold Change</option>
        </select>
      </div>

      <TaxaList
        significantTaxa={selectedCluster.significantTaxa}
        sortBy={sortBy}
      />
    </>
  );
};

const Sidebar: React.FC = () => {
  const selectedCluster = useSelectedCluster();
  const sidebarOpen = useSidebarOpen();

  if (!sidebarOpen) {
    return null;
  }

  return (
    <div id="sidebar">
      <h2>Cluster Information</h2>
      <div id="cluster-details">
        {selectedCluster ? (
          <ClusterDetails clusterId={selectedCluster.clusterId} />
        ) : (
          <p>Click on a cluster to see details.</p>
        )}
      </div>
    </div>
  );
};

export default Sidebar;
