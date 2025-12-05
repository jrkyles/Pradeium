import type { InputFieldMeta } from "../types/api";
import { PredictivePanel } from "./PredictivePanel";

type StorySectionProps = {
  stats: {
    medianLTV: string;
    medianDSCR: string;
    medianLoanSize: string;
    medianNoteRate: string;
    medianPropertyAge: string;
  };
  inputs: InputFieldMeta[];
  loading?: boolean;
  summaryError?: string | null;
};

export const StorySection = ({
  stats,
  inputs,
  loading,
  summaryError,
}: StorySectionProps) => (
  <section className="story-section" id="story">
    <div className="story-grid">
      <article className="story-panel">
        <p className="eyebrow">Our story</p>
        <h2>Multifamily delinquency intelligence, productized.</h2>
        <p>
          Praedium started as a USC research initiative studying how Fannie/Freddie
          multifamily collateral migrates into 60+ day delinquency. The platform
          now blends that research with design-grade UI so teams can test scenarios
          before credit committees ever meet.
        </p>
      </article>
      <article className="story-panel">
        <h3>Build process</h3>
        <p>
          Every feature—loan size, structure, occupancy, and property age—is
          cleaned, standardized, and benchmarked against historical delinquency
          outcomes. Weekly retrains keep pace with rate moves and IO burn-off.
        </p>
      </article>
      <article className="story-panel">
        <h3>Data sourcing</h3>
        <p>
          Agency performance tapes and servicer updates are cleaned into a shared
          schema. Median LTV ({stats.medianLTV}) and DSCR ({stats.medianDSCR}) frame
          credit posture while loan size ({stats.medianLoanSize}) and property age
          ({stats.medianPropertyAge}) calibrate the base case.
        </p>
      </article>
      <article className="story-panel stats">
        <div>
          <span>Median loan size</span>
          <strong>{stats.medianLoanSize}</strong>
        </div>
        <div>
          <span>Loan-to-value</span>
          <strong>{stats.medianLTV}</strong>
        </div>
        <div>
          <span>Debt service coverage</span>
          <strong>{stats.medianDSCR}</strong>
        </div>
        <div>
          <span>Note rate</span>
          <strong>{stats.medianNoteRate}</strong>
        </div>
        <div>
          <span>Property age</span>
          <strong>{stats.medianPropertyAge}</strong>
        </div>
      </article>
    </div>
    <div className="story-panel full">
      <p className="eyebrow">Ready when you are</p>
      <h3>Use the same predictive experience below.</h3>
      <PredictivePanel
        variant="inline"
        inputs={inputs}
        loading={loading}
        summaryError={summaryError}
      />
    </div>
  </section>
);
