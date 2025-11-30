import React from "react";
import ReactDOM from "react-dom/client";
import "bulma/css/bulma.min.css";

import App from "./components/App";

const rootElement = document.getElementById("root");
if (!rootElement) {
  throw new Error("Root element not found");
}

const root = ReactDOM.createRoot(rootElement);
root.render(<App />);
