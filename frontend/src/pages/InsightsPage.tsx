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
          <h2>From loan inputs to a defensible risk score</h2>
          <ol className="insights-steps">
            <li>
              <strong>1. From loan inputs to risk score</strong>
              <p>
                Praedium ingests the features lenders already track—DSCR, LTV,
                rate and structure, term, location, vintage, and recent performance.
                Inputs are validated, normalized, encoded, then passed to our
                trained XGBoost classifier to produce a probability of 60+ day
                delinquency. The UI translates that into a clear risk band.
              </p>
            </li>
            <li>
              <strong>2. Model performance in practice</strong>
              <p>
                On held-out data the XGBoost model materially beats the logistic
                baseline. It keeps recall high on delinquent loans while lifting
                precision, so most flagged loans truly deserve attention—fewer
                missed delinquencies and fewer wasted alerts.
              </p>
            </li>
            <li>
              <strong>3. How XGBoost works under the hood</strong>
              <p>
                XGBoost builds many shallow trees, each correcting the last using
                gradients of the loss. Splits target residual errors, then get
                scaled by a learning rate and added to the ensemble. Regularization
                keeps trees small, capturing nonlinear interactions (e.g., LTV ×
                market vacancy × vintage) without overfitting.
              </p>
            </li>
            <li>
              <strong>4. Why it matters for lenders</strong>
              <p>
                The ensemble adapts as rates, rents, and costs move—surfacing loans
                that look fine on static grids but share patterns with past
                delinquents. Users still see an intuitive probability and band
                while the boosted trees do the heavy lifting in the background.
              </p>
            </li>
          </ol>
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
