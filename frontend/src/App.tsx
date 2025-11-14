import { Route, Routes } from "react-router-dom";
import "./App.css";
import { LandingPage } from "./pages/LandingPage";
import { PredictionPage } from "./pages/PredictionPage";
import { InsightsPage } from "./pages/InsightsPage";

function App() {
  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route path="/predict" element={<PredictionPage />} />
      <Route path="/insights" element={<InsightsPage />} />
    </Routes>
  );
}

export default App;
