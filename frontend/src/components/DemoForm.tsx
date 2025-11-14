import { useEffect, useMemo, useState } from "react";
import type { FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import { fetchPrediction } from "../api";
import type { InputFieldMeta } from "../types/api";
import { usePrediction } from "../context/PredictionContext";

type DemoFormProps = {
  inputs: InputFieldMeta[];
  loading?: boolean;
  summaryError?: string | null;
};

export const DemoForm = ({
  inputs,
  loading = false,
  summaryError = null,
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
    setFormValues((prev) => {
      const next = { ...prev };
      let changed = false;
      sortedInputs.forEach((field) => {
        if (!next[field.key] || next[field.key] === "") {
          next[field.key] = field.median?.toString() ?? "";
          changed = true;
        }
      });
      return changed ? next : prev;
    });
  }, [sortedInputs]);

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    if (!sortedInputs.length) {
      setError("Inputs are not available yet.");
      return;
    }
    setError(null);
    const missing = sortedInputs.filter((field) => !formValues[field.key]);
    if (missing.length) {
      setError("Please complete all fields to run a projection.");
      return;
    }
    setIsSubmitting(true);
    try {
      const payload: Record<string, number> = {};
      sortedInputs.forEach((field) => {
        payload[field.key] = Number(
          (formValues[field.key] ?? "0").toString().replace(/,/g, "")
        );
      });
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
              <input
                type="number"
                step="any"
                value={formValues[input.key] ?? ""}
                placeholder={`${input.min} – ${input.max}`}
                onChange={(event) =>
                  setFormValues((prev) => ({
                    ...prev,
                    [input.key]: event.target.value,
                  }))
                }
              />
            </label>
          ))}
        </div>
      )}
      {error && <p className="form-error">{error}</p>}
      <button
        className="primary-btn"
        type="submit"
        disabled={isSubmitting || loading || !sortedInputs.length}
      >
        {isSubmitting ? "Predicting..." : "Predict"}
      </button>
    </form>
  );
};
