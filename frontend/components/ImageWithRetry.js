import React, { useState, useEffect, useRef } from "react";

const ImageWithRetry = ({ src, alt, style }) => {
  const [imageSrc, setImageSrc] = useState(null);
  const [error, setError] = useState(false);
  const [loading, setLoading] = useState(true);
  const retryCountRef = useRef(0);
  const isMountedRef = useRef(true);

  useEffect(() => {
    isMountedRef.current = true;

    if (!src) {
      setLoading(false);
      setError(true);
      return;
    }

    const maxRetries = 3;

    const attemptLoad = async (delayMs = 0) => {
      if (delayMs > 0) {
        await new Promise((resolve) => setTimeout(resolve, delayMs));
      }

      if (!isMountedRef.current) return;

      try {
        // Try HEAD request to check for rate limiting
        const headResponse = await fetch(src + "?width=50", {
          method: "HEAD",
        });

        if (!isMountedRef.current) return;

        if (!headResponse.ok) {
          if (headResponse.status === 429 || headResponse.status === 503) {
            const retryAfter = headResponse.headers.get("Retry-After");
            let delaySeconds = 5;

            if (retryAfter) {
              const retryAfterNum = parseInt(retryAfter, 10);
              if (!isNaN(retryAfterNum)) {
                delaySeconds = retryAfterNum;
              } else {
                const retryDate = new Date(retryAfter);
                const now = new Date();
                delaySeconds = Math.max(0, (retryDate - now) / 1000);
              }
            }

            delaySeconds = Math.min(delaySeconds, 30);

            if (retryCountRef.current < maxRetries) {
              retryCountRef.current++;
              console.log(
                `Rate limited on image ${alt}. Retrying in ${delaySeconds}s (attempt ${retryCountRef.current}/${maxRetries})`,
              );
              attemptLoad(delaySeconds * 1000);
              return;
            } else {
              throw new Error(
                `Rate limited after ${maxRetries} retries: ${headResponse.status}`,
              );
            }
          }

          throw new Error(`HTTP error ${headResponse.status}`);
        }

        // HEAD request succeeded, now load the actual image
        if (isMountedRef.current) {
          setImageSrc(src + "?width=50");
        }
      } catch (err) {
        if (!isMountedRef.current) return;

        console.error(`Failed to load image ${alt}:`, err);
        setError(true);
        setLoading(false);
      }
    };

    attemptLoad();

    return () => {
      isMountedRef.current = false;
    };
  }, [src, alt]);

  const handleImageLoad = () => {
    if (isMountedRef.current) {
      setLoading(false);
      setError(false);
    }
  };

  const handleImageError = () => {
    if (isMountedRef.current) {
      console.error(`Image tag failed to load: ${alt}`);
      setError(true);
      setLoading(false);
    }
  };

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
          ...
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
