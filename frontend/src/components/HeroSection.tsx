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
          <h1>Praedium</h1>
          <p className="hero-subtitle">
            Credit intelligence for the next era of real assets.
          </p>
        </div>
        <PredictivePanel
          inputs={inputs}
          loading={loading}
          summaryError={summaryError}
          variant="hero"
        />
      </div>
      <div className="hero-scroll-hint">Scroll to explore the thesis</div>
    </section>
  );
};
