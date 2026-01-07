import { useEffect, useMemo, useState } from "react";
import clsx from "clsx";
import { Link } from "react-router-dom";
import { ProbabilityChart } from "../components/ProbabilityChart";
import { usePrediction } from "../context/PredictionContext";
import { fetchSummary } from "../api";
import type { InputFieldMeta } from "../types/api";
import { PredictivePanel } from "../components/PredictivePanel";

export const PredictionPage = () => {
  const { prediction, lastPrediction } = usePrediction();
  const [inputs, setInputs] = useState<InputFieldMeta[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [panelOpen, setPanelOpen] = useState(false);
  const [compareMode, setCompareMode] = useState(false);

  useEffect(() => {
    if (!panelOpen || inputs.length) {
      return;
    }
    const loadInputs = async () => {
      try {
        setLoading(true);
        const summary = await fetchSummary();
        setInputs(summary.inputs ?? []);
        setError(null);
      } catch {
        setError("Unable to load inputs");
      } finally {
        setLoading(false);
      }
    };
    loadInputs();
  }, [panelOpen, inputs.length]);

  if (!prediction) {
    return (
      <section className="prediction-page empty">
        <h2>No projection yet</h2>
        <p>Start a scenario to see a projected grade and distribution.</p>
        <div className={clsx("prediction-inline-panel", panelOpen && "open")}>
          <PredictivePanel
            variant="inline"
            inputs={inputs}
            loading={loading}
            summaryError={error}
            inlineLabel="Run a scenario"
            autoOpen
            onRequestClose={() => setPanelOpen(false)}
            hideTrigger
          />
        </div>
      </section>
    );
  }

  const hasComparison = useMemo(
    () => Boolean(prediction && lastPrediction),
    [prediction, lastPrediction]
  );

  const betterIsCurrent =
    hasComparison &&
    (prediction?.probability ?? 1) <= (lastPrediction?.probability ?? 1);

  const comparisonData = useMemo(() => {
    if (!hasComparison || !prediction || !lastPrediction) return null;
    const categories = new Set<string>();
    prediction.distribution.forEach((p) => categories.add(p.category));
    lastPrediction.distribution.forEach((p) => categories.add(p.category));
    return Array.from(categories).map((category) => {
      const current = prediction.distribution.find((p) => p.category === category);
      const previous = lastPrediction.distribution.find((p) => p.category === category);
      return {
        category,
        current: current?.probability ?? 0,
        previous: previous?.probability ?? 0,
      };
    });
  }, [hasComparison, prediction, lastPrediction]);

  return (
    <section className="prediction-view">
      <div className={clsx("prediction-card", panelOpen && "expanded")}>
        <p className="eyebrow">Predicted letter grade</p>
        <h1>{prediction.rating}</h1>
        <p className="probability">
          {(prediction.probability * 100).toFixed(1)}% delinquency probability
        </p>
        <p>
          Grades are mapped from the probability of a 60+ day delinquency event.
          Use the distribution to see how risk shifts as you adjust structure
          and property inputs.
        </p>
        <div className="prediction-actions">
          <button
            className="scenario-btn"
            type="button"
            onClick={() => {
              setCompareMode(false);
              setPanelOpen((prev) => !prev);
            }}
          >
            {panelOpen ? "Hide input panel" : "Run another scenario"}
          </button>
          <button
            className="scenario-btn secondary"
            type="button"
            disabled={!prediction}
            onClick={() => {
              if (!prediction) return;
              setCompareMode(true);
              setPanelOpen(true);
            }}
          >
            Compare with new scenario
          </button>
        </div>
        <div
          className={clsx("prediction-inline-panel", panelOpen && "open")}
          aria-hidden={!panelOpen}
        >
          {panelOpen && (
            <PredictivePanel
              variant="inline"
              inputs={inputs}
              loading={loading}
              summaryError={error}
              inlineLabel="Enter scenario"
              autoOpen={panelOpen}
              onRequestClose={() => setPanelOpen(false)}
              hideTrigger
            />
          )}
        </div>
        <Link to="/insights" className="primary-btn secondary">
          How it works
        </Link>
      </div>
      <div className="prediction-chart-card">
        <div className="chart-header">
          <h3>Probability distribution</h3>
          {(compareMode || hasComparison) && (
            <p className="chart-caption">
              {hasComparison
                ? "Previous scenario shown in silver; current scenario in purple if it has the lower delinquency probability."
                : "Run a comparison scenario to overlay distributions."}
            </p>
          )}
        </div>
        {hasComparison && comparisonData ? (
          <ProbabilityChart
            data={prediction.distribution}
            comparison={{ data: comparisonData, betterIsCurrent }}
          />
        ) : (
          <ProbabilityChart data={prediction.distribution} />
        )}
      </div>
    </section>
  );
};
