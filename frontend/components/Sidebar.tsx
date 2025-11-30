import React, { useState, useMemo } from "react";
import ImageWithRetry from "./ImageWithRetry";
import {
  useSelectedCluster,
  useSidebarOpen,
  useClearSelection,
} from "../store/selectors";
import { type SignificantTaxa } from "../types";

interface TaxaItemProps {
  taxa: SignificantTaxa;
}

interface TaxaListProps {
  significantTaxa: TaxaItemProps["taxa"][];
  sortBy:
    | "count"
    | "high_log2_high_count_score"
    | "low_log2_high_count_score"
    | "foldchange";
}

interface ClusterDetailsProps {
  clusterId: number;
}

const TaxaItem: React.FC<TaxaItemProps> = ({ taxa }) => {
  return (
    <article className="media">
      <figure className="media-left">
        <figure className="image is-48x48" style={{ margin: 0 }}>
          {taxa.image_url ? (
            <ImageWithRetry
              src={taxa.image_url}
              alt={taxa.scientific_name}
              style={{ objectFit: "cover", width: "48px", height: "48px" }}
            />
          ) : (
            <div
              className="has-background-light"
              style={{ width: "48px", height: "48px" }}
            ></div>
          )}
        </figure>
      </figure>
      <div className="media-content">
        <div className="content">
          <p>
            <strong>{taxa.scientific_name}</strong>
            <br />
            <small className="has-text-grey">
              log2fc={taxa.log2_fold_change.toFixed(2)}, in-cluster=
              {taxa.cluster_count}, neighbors={taxa.neighbor_count}
            </small>
          </p>
        </div>
      </div>
    </article>
  );
};

const TaxaList: React.FC<TaxaListProps> = ({ significantTaxa, sortBy }) => {
  const sortedTaxa = useMemo(() => {
    if (significantTaxa.length === 0) return [];

    const taxa = [...significantTaxa];

    switch (sortBy) {
      case "count":
        return taxa.sort((a, b) => b.cluster_count - a.cluster_count);
      case "high_log2_high_count_score":
        return taxa.sort(
          (a, b) => b.high_log2_high_count_score - a.high_log2_high_count_score,
        );
      case "low_log2_high_count_score":
        return taxa.sort(
          (a, b) => b.low_log2_high_count_score - a.low_log2_high_count_score,
        );
      case "foldchange":
        return taxa.sort(
          (a, b) => Math.abs(b.log2_fold_change) - Math.abs(a.log2_fold_change),
        );
      default:
        return taxa;
    }
  }, [significantTaxa, sortBy]);

  if (significantTaxa.length === 0) {
    return (
      <p className="has-text-grey">No significant taxa for this cluster.</p>
    );
  }

  return (
    <>
      {sortedTaxa.map((taxa) => (
        <TaxaItem key={taxa.taxon_id} taxa={taxa} />
      ))}
    </>
  );
};

const ClusterDetails: React.FC<ClusterDetailsProps> = ({ clusterId }) => {
  const clearSelection = useClearSelection();
  const selectedCluster = useSelectedCluster();
  const [sortBy, setSortBy] = useState<
    | "count"
    | "high_log2_high_count_score"
    | "low_log2_high_count_score"
    | "foldchange"
  >("count");

  if (!selectedCluster) return null;

  return (
    <>
      <div className="level is-mobile mb-3">
        <div className="level-left">
          <div className="level-item">
            <h3 className="title is-5 mb-0">Cluster {clusterId}</h3>
          </div>
        </div>
        <div className="level-right">
          <div className="level-item">
            <button
              onClick={clearSelection}
              className="button is-danger is-small"
            >
              Clear
            </button>
          </div>
        </div>
      </div>

      <div className="field mb-4">
        <label htmlFor="sort-select" className="label is-small">
          Sort by:
        </label>
        <div className="control">
          <div className="select is-small is-fullwidth">
            <select
              id="sort-select"
              value={sortBy}
              onChange={(e) =>
                setSortBy(
                  e.target.value as
                    | "count"
                    | "high_log2_high_count_score"
                    | "low_log2_high_count_score"
                    | "foldchange",
                )
              }
            >
              <option value="count">Observation Count</option>
              <option value="high_log2_high_count_score">
                High Log2 High Count Score
              </option>
              <option value="low_log2_high_count_score">
                Low Log2 High Count Score
              </option>
              <option value="foldchange">Fold Change</option>
            </select>
          </div>
        </div>
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
    <div
      id="sidebar"
      className="box"
      style={{
        position: "absolute",
        top: 0,
        bottom: 0,
        right: 0,
        width: "30%",
        overflowY: "auto",
        borderRadius: 0,
      }}
    >
      <h2 className="title is-4">Cluster Information</h2>
      <div id="cluster-details">
        {selectedCluster ? (
          <ClusterDetails clusterId={selectedCluster.clusterId} />
        ) : (
          <p className="has-text-grey">Click on a cluster to see details.</p>
        )}
      </div>
    </div>
  );
};

export default Sidebar;
