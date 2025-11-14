import type { CSSProperties } from "react";
import type { InputFieldMeta } from "../types/api";
import { PredictivePanel } from "./PredictivePanel";

type HeroSectionProps = {
  fadeProgress: number;
  inputs: InputFieldMeta[];
  loading?: boolean;
  summaryError?: string | null;
};

export const HeroSection = ({
  fadeProgress,
  inputs,
  loading,
  summaryError,
}: HeroSectionProps) => {
  const heroStyle = {
    "--hero-fade": fadeProgress,
  } as CSSProperties;

  return (
    <section className="hero-section" style={heroStyle}>
      <div className="hero-overlay" aria-hidden="true" />
      <div className="hero-content">
        <div className="hero-title">
          <p className="hero-eyebrow">Praedium</p>
          <h1>Credit intelligence for the next era of real assets.</h1>
          <p className="hero-lede">
            One connected view of NOI, leverage, capital structure, and coverage
            so investment teams can defend a rating before stepping into the
            room.
          </p>
        </div>
        <PredictivePanel
          inputs={inputs}
          loading={loading}
          summaryError={summaryError}
          variant="hero"
          title="Demo the Praedium model"
          subtitle="Each field maps directly to the backend logistic regression engine."
        />
        <div className="hero-scroll-hint">Scroll to explore the thesis</div>
      </div>
    </section>
  );
};
