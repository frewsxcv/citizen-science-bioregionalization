import { create } from "zustand";
import { devtools } from "zustand/middleware";
import { SelectedCluster, ClusterData } from "../types";

interface AppState {
  // Cluster selection state
  selectedCluster: SelectedCluster | null;
  setSelectedCluster: (cluster: SelectedCluster | null) => void;

  // Map state
  mapInstance: maplibregl.Map | null;
  setMapInstance: (map: maplibregl.Map | null) => void;

  // UI state
  sidebarOpen: boolean;
  toggleSidebar: () => void;
  setSidebarOpen: (open: boolean) => void;

  // Data loading state
  isLoading: boolean;
  setIsLoading: (loading: boolean) => void;

  // Cluster data
  clusterData: ClusterData[];
  setClusterData: (data: ClusterData[]) => void;

  // Actions
  clearSelection: () => void;
  selectClusterById: (clusterId: number) => void;
}

export const useStore = create<AppState>()(
  devtools(
    (set, get) => ({
      // Initial state
      selectedCluster: null,
      mapInstance: null,
      sidebarOpen: true,
      isLoading: false,
      clusterData: [],

      // Setters
      setSelectedCluster: (cluster) => {
        set({ selectedCluster: cluster }, false, "setSelectedCluster");
      },

      setMapInstance: (map) => {
        set({ mapInstance: map }, false, "setMapInstance");
      },

      toggleSidebar: () => {
        set(
          (state) => ({ sidebarOpen: !state.sidebarOpen }),
          false,
          "toggleSidebar",
        );
      },

      setSidebarOpen: (open) => {
        set({ sidebarOpen: open }, false, "setSidebarOpen");
      },

      setIsLoading: (loading) => {
        set({ isLoading: loading }, false, "setIsLoading");
      },

      setClusterData: (data) => {
        set({ clusterData: data }, false, "setClusterData");
      },

      // Actions
      clearSelection: () => {
        set({ selectedCluster: null }, false, "clearSelection");
      },

      selectClusterById: (clusterId) => {
        const { clusterData } = get();
        const cluster = clusterData.find((c) => c.cluster === clusterId);

        if (cluster) {
          set(
            {
              selectedCluster: {
                clusterId: cluster.cluster,
                significantTaxa: cluster.significant_taxa,
              },
            },
            false,
            "selectClusterById",
          );
        }
      },
    }),
    {
      name: "BioregionalizationStore",
    },
  ),
);
