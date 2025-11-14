import { Link } from "react-router-dom";
import { ProbabilityChart } from "../components/ProbabilityChart";
import { usePrediction } from "../context/PredictionContext";

export const PredictionPage = () => {
  const { prediction } = usePrediction();

  if (!prediction) {
    return (
      <section className="prediction-page empty">
        <h2>No projection yet</h2>
        <p>Start from the landing page to submit financials and run Praedium.</p>
        <Link to="/" className="primary-btn">
          Back to landing
        </Link>
      </section>
    );
  }

  return (
    <section className="prediction-view">
      <div className="prediction-card">
        <p className="eyebrow">Predicted letter grade</p>
        <h1>{prediction.rating}</h1>
        <p className="probability">
          {(prediction.probability * 100).toFixed(1)}% confidence
        </p>
        <p>
          Praedium surfaces the most probable outcome while allowing you to step
          through the full probability distribution on the right.
        </p>
        <div className="prediction-actions">
          <Link to="/" className="link-btn">
            Run another scenario
          </Link>
          <Link to="/insights" className="primary-btn secondary">
            View feature correlations
          </Link>
        </div>
      </div>
      <div className="prediction-chart-card">
        <h3>Probability distribution</h3>
        <ProbabilityChart data={prediction.distribution} />
      </div>
    </section>
  );
};

