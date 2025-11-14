import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import "./index.css";
import App from "./App";
import { PredictionProvider } from "./context/PredictionContext";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BrowserRouter>
      <PredictionProvider>
        <App />
      </PredictionProvider>
    </BrowserRouter>
  </StrictMode>
);
