type QueueItem = {
  src: string;
  resolve: (success: boolean) => void;
  reject: (error: Error) => void;
};

class ImageLoadQueue {
  private queue: QueueItem[] = [];
  private isProcessing: boolean = false;
  private delayBetweenLoads: number = 100; // ms delay between image loads

  async loadImage(src: string): Promise<boolean> {
    return new Promise((resolve, reject) => {
      this.queue.push({ src, resolve, reject });
      this.processQueue();
    });
  }

  private async processQueue() {
    if (this.isProcessing || this.queue.length === 0) {
      return;
    }

    this.isProcessing = true;

    while (this.queue.length > 0) {
      const item = this.queue.shift();
      if (!item) break;

      try {
        const success = await this.loadSingleImage(item.src);
        item.resolve(success);
      } catch (error) {
        item.reject(error as Error);
      }

      // Small delay between loads to be nice to the server
      if (this.queue.length > 0) {
        await new Promise((resolve) =>
          setTimeout(resolve, this.delayBetweenLoads),
        );
      }
    }

    this.isProcessing = false;
  }

  private loadSingleImage(src: string): Promise<boolean> {
    return new Promise((resolve) => {
      const img = new Image();

      img.onload = () => {
        resolve(true);
      };

      img.onerror = () => {
        resolve(false);
      };

      img.src = src;
    });
  }

  setDelay(delayMs: number) {
    this.delayBetweenLoads = delayMs;
  }

  clear() {
    // Reject all pending promises with a cancellation error
    while (this.queue.length > 0) {
      const item = this.queue.shift();
      if (item) {
        item.reject(new Error("Queue cleared - new cluster selected"));
      }
    }
  }

  getQueueLength(): number {
    return this.queue.length;
  }
}

// Export a singleton instance
export const imageLoadQueue = new ImageLoadQueue();
