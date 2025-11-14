import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { fetchCorrelations } from "../api";

export const InsightsPage = () => {
  const [chart, setChart] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadChart = async () => {
      try {
        const data = await fetchCorrelations();
        setChart(data.chart);
      } catch {
        setError("Unable to load correlation insights.");
      }
    };
    loadChart();
  }, []);

  return (
    <section className="insights-page">
      <header>
        <h2>Feature correlations</h2>
        <p>
          Dive deeper into the variables that most influence Praedium’s credit
          view.
        </p>
      </header>
      {error && <p className="form-error">{error}</p>}
      {chart ? (
        <img
          src={`data:image/png;base64,${chart}`}
          alt="Praedium correlation plots"
        />
      ) : (
        !error && <p className="status-text">Loading visuals…</p>
      )}
      <Link to="/" className="primary-btn secondary">
        Back to landing
      </Link>
    </section>
  );
};


