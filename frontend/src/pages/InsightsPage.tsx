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
    <section className="insights-view">
      <header>
        <h2>Feature correlations</h2>
        <p>
          See how each operating input contributes to Praedium’s credit view.
          We render the backend correlation plots so your team never works from
          a black box.
        </p>
      </header>
      {error && <p className="form-error">{error}</p>}
      <div className="insights-card">
        {chart ? (
          <img
            src={`data:image/png;base64,${chart}`}
            alt="Praedium correlation plots"
          />
        ) : (
          !error && <p className="status-text">Loading visuals…</p>
        )}
      </div>
      <Link to="/" className="primary-btn secondary">
        Back to landing
      </Link>
    </section>
  );
};

