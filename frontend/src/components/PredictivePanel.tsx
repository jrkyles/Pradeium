import { memo, useCallback, useEffect, useState } from "react";
import clsx from "clsx";
import type { InputFieldMeta } from "../types/api";
import { DemoForm } from "./DemoForm";

type PredictivePanelProps = {
  inputs: InputFieldMeta[];
  loading?: boolean;
  summaryError?: string | null;
  variant?: "hero" | "inline";
  inlineLabel?: string;
  autoOpen?: boolean;
  onRequestClose?: () => void;
  hideTrigger?: boolean;
};

const PredictivePanelComponent = ({
  inputs,
  loading,
  summaryError,
  variant = "hero",
  inlineLabel = "Demo the model",
  autoOpen = false,
  onRequestClose,
  hideTrigger = false,
}: PredictivePanelProps) => {
  const [open, setOpen] = useState(autoOpen);

  useEffect(() => {
    setOpen(autoOpen);
  }, [autoOpen]);

  const handleOpen = useCallback(() => setOpen(true), []);
  const handleClose = useCallback(() => {
    setOpen(false);
    onRequestClose?.();
  }, [onRequestClose]);

  return (
    <div
      className={clsx(
        "predictive-panel",
        `predictive-panel--${variant}`,
        open && "open"
      )}
      data-open={open}
    >
      {!hideTrigger && (
        <button
          type="button"
          className="panel-cta"
          onClick={handleOpen}
          aria-expanded={open}
          data-visible={!open}
        >
          <span>{inlineLabel}</span>
          <small>Experience the live model</small>
        </button>
      )}
      <div className="panel-content" data-visible={open}>
        <DemoForm
          inputs={inputs}
          loading={loading}
          summaryError={summaryError}
          onBack={handleClose}
        />
      </div>
    </div>
  );
};

export const PredictivePanel = memo(PredictivePanelComponent);
