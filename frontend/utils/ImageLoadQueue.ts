import { useStore } from "../store/useStore";

/**
 * Hook to access image load queue functionality.
 * This replaces the previous singleton implementation with Zustand state management.
 *
 * @example
 * const { loadImage, clearImageQueue, getQueueLength, setDelayBetweenLoads } = useImageLoadQueue();
 *
 * // Load an image
 * const success = await loadImage('/path/to/image.jpg');
 *
 * // Clear the queue (e.g., when switching clusters)
 * clearImageQueue();
 */
export const useImageLoadQueue = () => {
  const loadImage = useStore((state) => state.loadImage);
  const clearImageQueue = useStore((state) => state.clearImageQueue);
  const getQueueLength = useStore((state) => state.getQueueLength);
  const setDelayBetweenLoads = useStore((state) => state.setDelayBetweenLoads);
  const queueLength = useStore((state) => state.imageQueue.length);
  const isProcessing = useStore((state) => state.isProcessingQueue);

  return {
    loadImage,
    clearImageQueue,
    getQueueLength,
    setDelayBetweenLoads,
    queueLength,
    isProcessing,
  };
};

/**
 * For non-React contexts (e.g., utility functions), you can access the queue directly:
 *
 * @example
 * import { useStore } from "../store/useStore";
 *
 * // Outside of React components
 * const success = await useStore.getState().loadImage('/path/to/image.jpg');
 * useStore.getState().clearImageQueue();
 */
