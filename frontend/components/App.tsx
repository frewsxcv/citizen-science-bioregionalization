import React from "react";
import Map from "./Map";
import Sidebar from "./Sidebar";
import { useSidebarOpen, useToggleSidebar } from "../store/selectors";

const App: React.FC = () => {
  const sidebarOpen = useSidebarOpen();
  const toggleSidebar = useToggleSidebar();

  return (
    <>
      <Map />
      <Sidebar />
      {/* Optional: Add a toggle button for the sidebar */}
      <button
        onClick={toggleSidebar}
        style={{
          position: "absolute",
          top: "10px",
          right: "10px",
          zIndex: 1000,
          padding: "8px 12px",
          background: "white",
          border: "1px solid #ccc",
          borderRadius: "4px",
          cursor: "pointer",
        }}
      >
        {sidebarOpen ? "Hide" : "Show"} Sidebar
      </button>
    </>
  );
};

export default App;
