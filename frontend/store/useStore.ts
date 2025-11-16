import { create } from "zustand";
import { devtools } from "zustand/middleware";
import { SelectedCluster, ClusterData } from "../types";

type QueueItem = {
  src: string;
  resolve: (success: boolean) => void;
  reject: (error: Error) => void;
};

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

  // Image load queue state
  imageQueue: QueueItem[];
  isProcessingQueue: boolean;
  delayBetweenLoads: number;

  // Image load queue actions
  loadImage: (src: string) => Promise<boolean>;
  setDelayBetweenLoads: (delayMs: number) => void;
  clearImageQueue: () => void;
  getQueueLength: () => number;

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
      imageQueue: [],
      isProcessingQueue: false,
      delayBetweenLoads: 100,

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

      // Image load queue actions
      loadImage: async (src) => {
        return new Promise((resolve, reject) => {
          set(
            (state) => ({
              imageQueue: [...state.imageQueue, { src, resolve, reject }],
            }),
            false,
            "loadImage:enqueue",
          );

          // Trigger queue processing
          const processQueue = async () => {
            const state = get();
            if (state.isProcessingQueue || state.imageQueue.length === 0) {
              return;
            }

            set(
              { isProcessingQueue: true },
              false,
              "loadImage:startProcessing",
            );

            while (get().imageQueue.length > 0) {
              const currentState = get();
              const item = currentState.imageQueue[0];
              if (!item) break;

              // Remove item from queue
              set(
                (state) => ({
                  imageQueue: state.imageQueue.slice(1),
                }),
                false,
                "loadImage:dequeue",
              );

              try {
                const success = await new Promise<boolean>((resolveLoad) => {
                  const img = new Image();
                  img.onload = () => resolveLoad(true);
                  img.onerror = () => resolveLoad(false);
                  img.src = item.src;
                });
                item.resolve(success);
              } catch (error) {
                item.reject(error as Error);
              }

              // Small delay between loads to be nice to the server
              if (get().imageQueue.length > 0) {
                await new Promise((resolveDelay) =>
                  setTimeout(resolveDelay, get().delayBetweenLoads),
                );
              }
            }

            set(
              { isProcessingQueue: false },
              false,
              "loadImage:stopProcessing",
            );
          };

          processQueue();
        });
      },

      setDelayBetweenLoads: (delayMs) => {
        set({ delayBetweenLoads: delayMs }, false, "setDelayBetweenLoads");
      },

      clearImageQueue: () => {
        const { imageQueue } = get();
        // Reject all pending promises with a cancellation error
        imageQueue.forEach((item) => {
          item.reject(new Error("Queue cleared - new cluster selected"));
        });
        set({ imageQueue: [] }, false, "clearImageQueue");
      },

      getQueueLength: () => {
        return get().imageQueue.length;
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
