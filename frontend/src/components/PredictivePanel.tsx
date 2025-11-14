import { useState } from "react";
import clsx from "clsx";
import type { InputFieldMeta } from "../types/api";
import { DemoForm } from "./DemoForm";

type PredictivePanelProps = {
  inputs: InputFieldMeta[];
  loading?: boolean;
  summaryError?: string | null;
  variant?: "hero" | "inline";
  title?: string;
  subtitle?: string;
};

export const PredictivePanel = ({
  inputs,
  loading,
  summaryError,
  variant = "hero",
  title = "Run a predictive scenario",
  subtitle = "Enter the operating profile to see our credit view instantly.",
}: PredictivePanelProps) => {
  const [open, setOpen] = useState(false);

  return (
    <div
      className={clsx(
        "predictive-panel",
        `predictive-panel--${variant}`,
        open && "open"
      )}
    >
      <button
        type="button"
        className="panel-cta"
        onClick={() => setOpen(true)}
        aria-expanded={open}
      >
        <span>Demo the model</span>
        <small>Experience the live model</small>
      </button>
      <div className="panel-content">
        <div className="panel-heading">
          <div>
            <p className="eyebrow">Praedium Engine</p>
            <h3>{title}</h3>
            {subtitle && <p>{subtitle}</p>}
          </div>
          <button
            type="button"
            className="ghost-btn"
            onClick={() => setOpen(false)}
          >
            Collapse
          </button>
        </div>
        <DemoForm
          inputs={inputs}
          loading={loading}
          summaryError={summaryError}
        />
      </div>
    </div>
  );
};

