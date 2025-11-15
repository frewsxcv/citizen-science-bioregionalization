import { useStore } from "./useStore";
import { imageLoadQueue } from "../utils/ImageLoadQueue";
import { ClusterData, SelectedCluster } from "../types";

/**
 * Complex actions and side effects for the store
 * These are functions that can be called from components to trigger
 * multiple state updates or async operations
 */

/**
 * Select a cluster and preload its images
 */
export const selectClusterWithImagePreload = (cluster: SelectedCluster) => {
  const { setSelectedCluster } = useStore.getState();

  // Clear the image queue to prioritize new cluster's images
  imageLoadQueue.clear();

  // Set the selected cluster
  setSelectedCluster(cluster);

  // Preload images for the selected cluster
  const taxaWithImages = cluster.significantTaxa.filter(
    (taxa) => taxa.image_url !== null,
  );

  taxaWithImages.forEach((taxa) => {
    if (taxa.image_url) {
      imageLoadQueue.loadImage(taxa.image_url).catch((error) => {
        console.warn(
          `Failed to preload image for ${taxa.scientific_name}:`,
          error,
        );
      });
    }
  });
};

/**
 * Load cluster data with loading state management
 */
export const loadClusterData = async (dataUrl: string) => {
  const { setIsLoading, setClusterData } = useStore.getState();

  setIsLoading(true);

  try {
    const response = await fetch(dataUrl);
    if (!response.ok) {
      throw new Error(`Failed to fetch cluster data: ${response.statusText}`);
    }

    const data: ClusterData[] = await response.json();
    setClusterData(data);
  } catch (error) {
    console.error("Error loading cluster data:", error);
    throw error;
  } finally {
    setIsLoading(false);
  }
};

/**
 * Navigate to a specific cluster on the map
 */
export const flyToCluster = (clusterId: number) => {
  const { mapInstance, clusterData, selectClusterById } = useStore.getState();

  if (!mapInstance) {
    console.warn("Map instance not available");
    return;
  }

  const cluster = clusterData.find((c) => c.cluster === clusterId);
  if (!cluster) {
    console.warn(`Cluster ${clusterId} not found`);
    return;
  }

  // Get the bounds of the cluster polygon
  const coordinates = cluster.boundary.coordinates[0];
  if (!coordinates || coordinates.length === 0) {
    return;
  }

  // Calculate center of the polygon
  const lngs = coordinates.map((coord) => coord[0]);
  const lats = coordinates.map((coord) => coord[1]);

  const centerLng = lngs.reduce((a, b) => a + b, 0) / lngs.length;
  const centerLat = lats.reduce((a, b) => a + b, 0) / lats.length;

  // Fly to the cluster
  mapInstance.flyTo({
    center: [centerLng, centerLat],
    zoom: 8,
    duration: 1500,
  });

  // Select the cluster
  selectClusterById(clusterId);
};

/**
 * Export cluster data to JSON
 */
export const exportClusterData = (clusterId: number) => {
  const { clusterData } = useStore.getState();

  const cluster = clusterData.find((c) => c.cluster === clusterId);
  if (!cluster) {
    console.warn(`Cluster ${clusterId} not found`);
    return;
  }

  const dataStr = JSON.stringify(cluster, null, 2);
  const dataBlob = new Blob([dataStr], { type: "application/json" });
  const url = URL.createObjectURL(dataBlob);

  const link = document.createElement("a");
  link.href = url;
  link.download = `cluster-${clusterId}-data.json`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};

/**
 * Copy taxa list to clipboard
 */
export const copyTaxaToClipboard = async (clusterId?: number) => {
  const { selectedCluster, clusterData } = useStore.getState();

  let taxa = selectedCluster?.significantTaxa;

  // If clusterId provided, get taxa from that cluster
  if (clusterId !== undefined) {
    const cluster = clusterData.find((c) => c.cluster === clusterId);
    taxa = cluster?.significant_taxa;
  }

  if (!taxa || taxa.length === 0) {
    console.warn("No taxa to copy");
    return false;
  }

  // Format taxa as text
  const taxaText = taxa
    .map(
      (t) =>
        `${t.scientific_name} (p=${t.p_value.toExponential(2)}, log2fc=${t.log2_fold_change.toFixed(2)})`,
    )
    .join("\n");

  try {
    await navigator.clipboard.writeText(taxaText);
    return true;
  } catch (error) {
    console.error("Failed to copy to clipboard:", error);
    return false;
  }
};

/**
 * Reset all state to initial values
 */
export const resetApp = () => {
  const { clearSelection, setSidebarOpen, setIsLoading } = useStore.getState();

  clearSelection();
  setSidebarOpen(true);
  setIsLoading(false);
  imageLoadQueue.clear();
};

/**
 * Batch select multiple clusters
 * (Future feature - could show multiple clusters simultaneously)
 */
export const selectMultipleClusters = (clusterIds: number[]) => {
  const { clusterData } = useStore.getState();

  const clusters = clusterData.filter((c) => clusterIds.includes(c.cluster));

  if (clusters.length === 0) {
    console.warn("No matching clusters found");
    return;
  }

  // For now, just select the first one
  // In the future, this could be expanded to support multi-selection
  const firstCluster = clusters[0];
  const { setSelectedCluster } = useStore.getState();

  setSelectedCluster({
    clusterId: firstCluster.cluster,
    significantTaxa: firstCluster.significant_taxa,
  });
};

/**
 * Get store state snapshot for debugging
 */
export const getStateSnapshot = () => {
  const state = useStore.getState();

  return {
    selectedClusterId: state.selectedCluster?.clusterId ?? null,
    taxaCount: state.selectedCluster?.significantTaxa.length ?? 0,
    totalClusters: state.clusterData.length,
    sidebarOpen: state.sidebarOpen,
    isLoading: state.isLoading,
    hasMapInstance: state.mapInstance !== null,
  };
};

/**
 * Log current state to console (for debugging)
 */
export const debugState = () => {
  const snapshot = getStateSnapshot();
  console.log("🔍 Store State Snapshot:", snapshot);
  console.log("📦 Full State:", useStore.getState());
};

// Make debug functions available globally in development
if (typeof window !== "undefined") {
  (window as any).__DEBUG_STORE__ = {
    getState: () => useStore.getState(),
    getSnapshot: getStateSnapshot,
    debug: debugState,
    flyToCluster,
    exportClusterData,
    resetApp,
  };
}
