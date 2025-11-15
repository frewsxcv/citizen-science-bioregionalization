declare module maplibregl {
  export class Map {
    constructor(options: {
      container: HTMLElement;
      style: string;
      center: [number, number];
      zoom: number;
    });

    on(event: string, callback: (e?: any) => void): void;
    on(event: string, layer: string, callback: (e: any) => void): void;

    addSource(id: string, source: {
      type: string;
      data: GeoJSON.FeatureCollection;
    }): void;

    addLayer(layer: {
      id: string;
      type: string;
      source: string;
      paint: Record<string, any>;
    }): void;

    getCanvas(): HTMLCanvasElement;
  }
}

declare const maplibregl: typeof maplibregl;
