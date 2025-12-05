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
      <div className="insights-layout">
        <div className="insights-copy">
          <p className="eyebrow">How we predict</p>
          <h2>The mechanics behind Praedium</h2>
          <p>
            Praedium standardizes loan attributes from 50k+ agency multifamily
            records, encodes structure and geography, and feeds them into a
            class-balanced logistic model that predicts the probability of a 60+
            day delinquency. That PD is then mapped into A/B/C grades for quick
            credit triage.
          </p>
          <p>
            The plots on the right keep the process transparent. Strip plots show
            how continuous features shift between clean and delinquent loans,
            while bars highlight default rates by categorical attributes like
            structure, property type, and state.
          </p>
        </div>
        <div className="insights-visuals">
          {error && <p className="form-error">{error}</p>}
          {chart ? (
            <img
              src={`data:image/png;base64,${chart}`}
              alt="Praedium correlation plots"
            />
          ) : (
            !error && <p className="status-text">Loading visuals…</p>
          )}
        </div>
      </div>
      <Link to="/" className="primary-btn secondary">
        Back to landing
      </Link>
    </section>
  );
};
