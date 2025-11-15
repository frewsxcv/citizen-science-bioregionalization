import { useStore } from "./useStore";
import { SignificantTaxa } from "../types";

/**
 * Selectors for computed state values
 * These provide memoized access to derived data from the store
 */

// Cluster selection selectors
export const useSelectedCluster = () =>
  useStore((state) => state.selectedCluster);

export const useSelectedClusterId = () =>
  useStore((state) => state.selectedCluster?.clusterId ?? null);

export const useSelectedTaxa = () =>
  useStore((state) => state.selectedCluster?.significantTaxa ?? []);

// Check if a specific cluster is selected
export const useIsClusterSelected = (clusterId: number) =>
  useStore((state) => state.selectedCluster?.clusterId === clusterId);

// UI selectors
export const useSidebarOpen = () => useStore((state) => state.sidebarOpen);

export const useIsLoading = () => useStore((state) => state.isLoading);

// Map selectors
export const useMapInstance = () => useStore((state) => state.mapInstance);

// Cluster data selectors
export const useClusterData = () => useStore((state) => state.clusterData);

export const useClusterCount = () =>
  useStore((state) => state.clusterData.length);

export const useClusterById = (clusterId: number) =>
  useStore((state) =>
    state.clusterData.find((cluster) => cluster.cluster === clusterId),
  );

// Action selectors - Fixed to avoid infinite loops
export const useSetSelectedCluster = () =>
  useStore((state) => state.setSelectedCluster);

export const useClearSelection = () =>
  useStore((state) => state.clearSelection);

export const useSelectClusterById = () =>
  useStore((state) => state.selectClusterById);

export const useToggleSidebar = () => useStore((state) => state.toggleSidebar);

export const useSetSidebarOpen = () =>
  useStore((state) => state.setSidebarOpen);

export const useSetIsLoading = () => useStore((state) => state.setIsLoading);

// Computed selectors
export const useHasSelection = () =>
  useStore((state) => state.selectedCluster !== null);

// Sort taxa by different criteria
// NOTE: These create new arrays on each call, so they should be used with caution
// or replaced with useMemo in the component
export const useSortedTaxaByCount = () =>
  useStore((state) => {
    if (!state.selectedCluster) return [];
    return [...state.selectedCluster.significantTaxa].sort(
      (a, b) => b.cluster_count - a.cluster_count,
    );
  });

export const useSortedTaxaByPValue = () =>
  useStore((state) => {
    if (!state.selectedCluster) return [];
    return [...state.selectedCluster.significantTaxa].sort(
      (a, b) => a.p_value - b.p_value,
    );
  });

export const useSortedTaxaByFoldChange = () =>
  useStore((state) => {
    if (!state.selectedCluster) return [];
    return [...state.selectedCluster.significantTaxa].sort(
      (a, b) => Math.abs(b.log2_fold_change) - Math.abs(a.log2_fold_change),
    );
  });

// Get taxa with images
export const useTaxaWithImages = () =>
  useStore((state) => {
    if (!state.selectedCluster) return [];
    return state.selectedCluster.significantTaxa.filter(
      (taxa) => taxa.image_url !== null,
    );
  });

// Statistics selectors
// Returns individual stats values to avoid creating new objects
export const useTaxaCount = () =>
  useStore((state) => state.selectedCluster?.significantTaxa.length ?? 0);

export const useTotalObservations = () =>
  useStore(
    (state) =>
      state.selectedCluster?.significantTaxa.reduce(
        (sum, t) => sum + t.cluster_count,
        0,
      ) ?? 0,
  );

export const useAveragePValue = () =>
  useStore((state) => {
    if (!state.selectedCluster) return 0;
    const taxa = state.selectedCluster.significantTaxa;
    return taxa.reduce((sum, t) => sum + t.p_value, 0) / taxa.length;
  });

export const useAverageFoldChange = () =>
  useStore((state) => {
    if (!state.selectedCluster) return 0;
    const taxa = state.selectedCluster.significantTaxa;
    return taxa.reduce((sum, t) => sum + t.log2_fold_change, 0) / taxa.length;
  });

export const useTaxaWithImagesCount = () =>
  useStore(
    (state) =>
      state.selectedCluster?.significantTaxa.filter((t) => t.image_url !== null)
        .length ?? 0,
  );

export const useTaxaWithoutImagesCount = () =>
  useStore((state) => {
    if (!state.selectedCluster) return 0;
    const total = state.selectedCluster.significantTaxa.length;
    const withImages = state.selectedCluster.significantTaxa.filter(
      (t) => t.image_url !== null,
    ).length;
    return total - withImages;
  });

// Legacy selector that returns an object - USE INDIVIDUAL SELECTORS ABOVE INSTEAD
// This can cause infinite re-renders because it returns a new object each time
export const useClusterStats = () => {
  const totalTaxaCount = useTaxaCount();
  const totalObservations = useTotalObservations();
  const averagePValue = useAveragePValue();
  const averageFoldChange = useAverageFoldChange();
  const taxaWithImages = useTaxaWithImagesCount();
  const taxaWithoutImages = useTaxaWithoutImagesCount();

  const hasSelection = useStore((state) => state.selectedCluster !== null);

  if (!hasSelection) return null;

  return {
    totalTaxaCount,
    totalObservations,
    averagePValue,
    averageFoldChange,
    taxaWithImages,
    taxaWithoutImages,
  };
};
