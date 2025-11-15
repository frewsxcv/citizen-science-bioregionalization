import React, { useState, useRef, CSSProperties } from "react";
import { imageLoadQueue } from "../utils/ImageLoadQueue";

interface ImageWithRetryProps {
  src: string | null;
  alt: string;
  style?: CSSProperties;
}

const ImageWithRetry: React.FC<ImageWithRetryProps> = ({ src, alt, style }) => {
  const [error, setError] = useState(false);
  const [loading, setLoading] = useState(true);
  const [queued, setQueued] = useState(false);
  const retryCountRef = useRef(0);
  const retryTimeoutRef = useRef<number | null>(null);
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const attemptedLoadRef = useRef(false);

  const handleImageLoad = () => {
    setLoading(false);
    setQueued(false);
    setError(false);
    retryCountRef.current = 0;
  };

  const handleImageError = () => {
    const maxRetries = 3;

    if (retryCountRef.current < maxRetries) {
      retryCountRef.current++;
      const delaySeconds = Math.min(5 * retryCountRef.current, 30);

      console.log(
        `Failed to load image ${alt}. Retrying in ${delaySeconds}s (attempt ${retryCountRef.current}/${maxRetries})`,
      );

      setLoading(true);
      setQueued(false);

      // Retry with exponential backoff
      retryTimeoutRef.current = setTimeout(() => {
        if (src) {
          attemptLoadImage(src);
        }
      }, delaySeconds * 1000);
    } else {
      console.error(`Image failed to load after ${maxRetries} retries: ${alt}`);
      setError(true);
      setLoading(false);
      setQueued(false);
    }
  };

  const attemptLoadImage = React.useCallback(
    async (imageSrc: string) => {
      const fullSrc = imageSrc + `?width=50&retry=${retryCountRef.current}`;

      setQueued(true);
      setLoading(true);

      try {
        // Queue the image load
        await imageLoadQueue.loadImage(fullSrc);
        // Image loaded successfully, now set it to display
        setImageSrc(fullSrc);
      } catch (error) {
        console.error(`Failed to queue image ${alt}:`, error);
        setError(true);
        setLoading(false);
        setQueued(false);
      }
    },
    [alt],
  );

  // Cleanup timeout on unmount
  React.useEffect(() => {
    return () => {
      if (retryTimeoutRef.current) {
        clearTimeout(retryTimeoutRef.current);
      }
    };
  }, []);

  // Update image source when src prop changes
  React.useEffect(() => {
    retryCountRef.current = 0;
    attemptedLoadRef.current = false;
    if (retryTimeoutRef.current) {
      clearTimeout(retryTimeoutRef.current);
    }
    if (src) {
      setImageSrc(null);
      setLoading(true);
      setError(false);
      // Call async function without awaiting
      attemptLoadImage(src);
    } else {
      setImageSrc(null);
      setLoading(false);
      setError(true);
      setQueued(false);
    }
  }, [src, attemptLoadImage]);

  if (!src || error) {
    return (
      <div
        style={{
          ...style,
          background: "#eee",
        }}
      ></div>
    );
  }

  return (
    <>
      {loading && (
        <div
          style={{
            ...style,
            background: "#f0f0f0",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: "10px",
            color: "#999",
            position: "absolute",
          }}
        >
          {queued ? "queued..." : "..."}
        </div>
      )}
      {imageSrc && (
        <img
          src={imageSrc}
          alt={alt}
          style={{
            ...style,
            visibility: loading ? "hidden" : "visible",
          }}
          onLoad={handleImageLoad}
          onError={handleImageError}
        />
      )}
    </>
  );
};

export default ImageWithRetry;
