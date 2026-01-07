import { memo, useEffect, useMemo, useState } from "react";
import type { FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import { fetchPrediction } from "../api";
import type { InputFieldMeta } from "../types/api";
import { usePrediction } from "../context/PredictionContext";

type DemoFormProps = {
  inputs: InputFieldMeta[];
  loading?: boolean;
  summaryError?: string | null;
  onBack?: () => void;
  randomizeKey?: number;
};

const DemoFormComponent = ({
  inputs,
  loading = false,
  summaryError = null,
  onBack,
  randomizeKey = 0,
}: DemoFormProps) => {
  const [formValues, setFormValues] = useState<Record<string, string>>({});
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const navigate = useNavigate();
  const { setPrediction } = usePrediction();

  const sortedInputs = useMemo(
    () =>
      [...inputs].sort((a, b) =>
        a.label.localeCompare(b.label, undefined, { sensitivity: "base" })
      ),
    [inputs]
  );

  useEffect(() => {
    if (!sortedInputs.length) {
      return;
    }
    const next: Record<string, string> = {};
    const riskTierRoll = Math.random();
    const riskTier =
      riskTierRoll < 0.33 ? "low" : riskTierRoll < 0.66 ? "mid" : "high";

    const inverseRiskFields = new Set([
      "underwritten_dscr",
      "physical_occupancy",
    ]);

    sortedInputs.forEach((field) => {
      if (field.type === "numeric") {
        const min = field.min ?? 0;
        const max = field.max ?? field.median ?? min;
        const span = Math.max(max - min, 0);
        const median = field.median ?? (min + max) / 2;
        // bucket selection guided by risk tier
        let low: number;
        let high: number;
        const isInverse = inverseRiskFields.has(field.key);
        if (riskTier === "low") {
          low = isInverse ? median : min;
          high = isInverse ? max : min + span * 0.25;
        } else if (riskTier === "mid") {
          low = median - span * 0.1;
          high = median + span * 0.1;
        } else {
          low = isInverse ? min : max - span * 0.25;
          high = isInverse ? median : max;
        }
        if (high < low) {
          [low, high] = [high, low];
        }
        const sampled = low + Math.random() * Math.max(high - low, 0);
        // blend slightly toward median to avoid extreme tails
        const blended = (sampled * 0.7 + median * 0.3);
        next[field.key] = blended.toFixed(4);
      } else if (field.type === "categorical" && field.options?.length) {
        const opts = field.options;
        const randomIndex = Math.floor(Math.random() * opts.length);
        next[field.key] = opts[randomIndex];
      } else {
        next[field.key] = field.median?.toString() ?? "";
      }
    });
    setFormValues(next);
  }, [sortedInputs, randomizeKey]);

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    if (!sortedInputs.length) {
      setError("Inputs are not available yet.");
      return;
    }
    setError(null);
    const missing = sortedInputs.filter(
      (field) =>
        formValues[field.key] === undefined || formValues[field.key] === ""
    );
    if (missing.length) {
      setError("Please complete all fields to run a projection.");
      return;
    }
    setIsSubmitting(true);
    try {
      const payload: Record<string, string | number> = {};
      for (const field of sortedInputs) {
        const rawValue = formValues[field.key] ?? "";
        if (field.type === "numeric") {
          const numericValue = Number(rawValue.toString().replace(/,/g, ""));
          if (!Number.isFinite(numericValue)) {
            setError(`"${field.label}" must be a number.`);
            setIsSubmitting(false);
            return;
          }
          payload[field.key] = numericValue;
        } else {
          payload[field.key] = rawValue.toString();
        }
      }
      const result = await fetchPrediction(payload);
      setPrediction(result);
      navigate("/predict");
    } catch (err) {
      setError("We couldn’t run the projection. Please try again.");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <form className="demo-form" onSubmit={handleSubmit}>
      {loading && <p className="status-text">Loading inputs…</p>}
      {!loading && summaryError && (
        <p className="form-error">{summaryError}</p>
      )}
      {sortedInputs.length > 0 && (
        <div className="form-grid">
          {sortedInputs.map((input) => (
            <label key={input.key}>
              <span>{input.label}</span>
              {input.type === "numeric" ? (
                <input
                  type="number"
                  step="any"
                  value={formValues[input.key] ?? ""}
                  placeholder={
                    input.min !== undefined && input.max !== undefined
                      ? `${input.min} – ${input.max}`
                      : ""
                  }
                  onChange={(event) =>
                    setFormValues((prev) => ({
                      ...prev,
                      [input.key]: event.target.value,
                    }))
                  }
                />
              ) : (
                <select
                  value={formValues[input.key] ?? ""}
                  onChange={(event) =>
                    setFormValues((prev) => ({
                      ...prev,
                      [input.key]: event.target.value,
                    }))
                  }
                >
                  {(input.options ?? []).map((opt) => (
                    <option key={opt} value={opt}>
                      {opt}
                    </option>
                  ))}
                </select>
              )}
            </label>
          ))}
        </div>
      )}
      {error && <p className="form-error">{error}</p>}
      <div className="form-actions">
        {onBack && (
          <button
            type="button"
            className="outline-btn"
            onClick={onBack}
            disabled={isSubmitting}
          >
            Back
          </button>
        )}
        <button
          className="primary-btn"
          type="submit"
          disabled={isSubmitting || loading || !sortedInputs.length}
        >
          {isSubmitting ? "Predicting..." : "Predict"}
        </button>
      </div>
    </form>
  );
};

export const DemoForm = memo(DemoFormComponent);
