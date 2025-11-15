import React from "react";
import ImageWithRetry from "./ImageWithRetry";
import { SelectedCluster } from "../types";

interface SidebarProps {
  selectedCluster: SelectedCluster | null;
}

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
}

interface ClusterDetailsProps {
  clusterId: number;
  significantTaxa: TaxaItemProps["taxa"][];
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

const TaxaList: React.FC<TaxaListProps> = ({ significantTaxa }) => {
  if (significantTaxa.length === 0) {
    return <p>No significant taxa for this cluster.</p>;
  }

  const sortedTaxa = [...significantTaxa].sort(
    (a, b) => b.cluster_count - a.cluster_count,
  );

  return (
    <ul>
      {sortedTaxa.map((taxa) => (
        <TaxaItem key={taxa.taxon_id} taxa={taxa} />
      ))}
    </ul>
  );
};

const ClusterDetails: React.FC<ClusterDetailsProps> = ({
  clusterId,
  significantTaxa,
}) => {
  return (
    <>
      <h3>Cluster {clusterId}</h3>
      <TaxaList significantTaxa={significantTaxa} />
    </>
  );
};

const Sidebar: React.FC<SidebarProps> = ({ selectedCluster }) => {
  return (
    <div id="sidebar">
      <h2>Cluster Information</h2>
      <div id="cluster-details">
        {selectedCluster ? (
          <ClusterDetails
            clusterId={selectedCluster.clusterId}
            significantTaxa={selectedCluster.significantTaxa}
          />
        ) : (
          <p>Click on a cluster to see details.</p>
        )}
      </div>
    </div>
  );
};

export default Sidebar;
