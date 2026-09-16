import React from "react";
import { useAvailableLevels, useLevel, useSetLevel } from "../store/selectors";

/**
 * Chooses which cut of the merge tree is drawn.
 *
 * The levels are nested — a cluster at k=8 lies inside exactly one cluster at
 * k=4 — so this reads as zooming in on the regionalization rather than as
 * switching between unrelated maps.
 */
const LevelSelector: React.FC = () => {
  const levels = useAvailableLevels();
  const level = useLevel();
  const setLevel = useSetLevel();

  // A single level is the old behaviour and needs no control.
  if (levels.length < 2) return null;

  return (
    <div
      style={{
        position: "absolute",
        top: "10px",
        left: "10px",
        zIndex: 1000,
        background: "rgba(255, 255, 255, 0.92)",
        borderRadius: "4px",
        padding: "6px 8px",
        display: "flex",
        alignItems: "center",
        gap: "6px",
      }}
    >
      <span style={{ fontSize: "0.75rem", color: "#4a4a4a" }}>Regions</span>
      {levels.map((k) => (
        <button
          key={k}
          onClick={() => setLevel(k)}
          className={`button is-small ${k === level ? "is-info" : ""}`}
          aria-pressed={k === level}
          style={{ minWidth: "2.2rem" }}
        >
          {k}
        </button>
      ))}
    </div>
  );
};

export default LevelSelector;
