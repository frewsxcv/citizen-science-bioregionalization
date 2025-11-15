import React from "react";
import ImageWithRetry from "./ImageWithRetry";
import { SelectedCluster } from "../types";

interface SidebarProps {
  selectedCluster: SelectedCluster | null;
}

const Sidebar: React.FC<SidebarProps> = ({ selectedCluster }) => {
  return (
    <div id="sidebar">
      <h2>Cluster Information</h2>
      <div id="cluster-details">
        {selectedCluster ? (
          <>
            <h3>Cluster {selectedCluster.clusterId}</h3>
            {selectedCluster.significantTaxa.length > 0 ? (
              <ul>
                {selectedCluster.significantTaxa
                  .sort((a, b) => b.cluster_count - a.cluster_count)
                  .map((taxa) => (
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
                  ))}
              </ul>
            ) : (
              <p>No significant taxa for this cluster.</p>
            )}
          </>
        ) : (
          <p>Click on a cluster to see details.</p>
        )}
      </div>
    </div>
  );
};

export default Sidebar;
