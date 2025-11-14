import type { InputFieldMeta } from "../types/api";
import { PredictivePanel } from "./PredictivePanel";

type StorySectionProps = {
  stats: {
    medianNOI: string;
    medianInterestCoverage: string;
    medianLeverage: string;
    medianDefaultRate: string;
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
        <h2>Built with institutional research rigor.</h2>
        <p>
          Praedium started as a USC research initiative focused on explaining how
          operating statements translate to commercial mortgage outcomes. The
          platform now blends investment-banking detail with design-grade UI so
          teams can test conviction faster.
        </p>
      </article>
      <article className="story-panel">
        <h3>The build process</h3>
        <p>
          Every feature is engineered, standardized, and benchmarked against
          historical grade outcomes. We stress check the model weekly against
          market moves and publish explainers for each driver.
        </p>
      </article>
      <article className="story-panel">
        <h3>Data sourcing</h3>
        <p>
          Operating statements, sponsor submissions, and public market filings
          are cleaned into a shared schema. Median default rate ({stats.medianDefaultRate})
          anchors our loss assumptions while NOI ({stats.medianNOI}) and leverage
          ({stats.medianLeverage}x) set the credit posture.
        </p>
      </article>
      <article className="story-panel stats">
        <div>
          <span>Median NOI</span>
          <strong>{stats.medianNOI}</strong>
        </div>
        <div>
          <span>Interest coverage</span>
          <strong>{stats.medianInterestCoverage}x</strong>
        </div>
        <div>
          <span>Net leverage</span>
          <strong>{stats.medianLeverage}x</strong>
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
        title="Demo the model from this section"
        subtitle="The morphing panel mirrors the hero experience so teams can launch a scenario anywhere on the page."
      />
    </div>
  </section>
);
