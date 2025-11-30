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
      <button
        onClick={toggleSidebar}
        className="button is-small"
        style={{
          position: "absolute",
          top: "10px",
          right: "10px",
          zIndex: 1000,
        }}
      >
        {sidebarOpen ? "Hide" : "Show"} Sidebar
      </button>
    </>
  );
};

export default App;
