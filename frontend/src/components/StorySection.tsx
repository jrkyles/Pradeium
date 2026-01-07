import clsx from "clsx";
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

const thesisSections = [
  {
    title: "Identifying the delinquency pain point",
    body: [
      "Banks and lenders sit on trillions in multifamily exposure. Even small moves in delinquency translate into billions of balance-sheet risk as rates rise and rent growth softens.",
      "Risk teams told us the same story: signals show up in DSCR drift, LTV shifts, and market vacancies long before a loan hits 60+ days past due—but those signals live in scattered spreadsheets and static scorecards.",
      "Praedium frames delinquency as a forward signal, giving credit teams a daily, data-backed view instead of waiting for arrears to show up in aging reports.",
    ],
  },
  {
    title: "Why traditional metrics fall short",
    body: [
      "Point-in-time DSCR, LTV, and occupancy were built for a slow-moving world. Today, rate resets, NOI compression, and supply shocks can turn a “fine” loan into a problem in weeks.",
      "Manual sweeps and ad hoc stress tests miss emerging risk across the book. Praedium treats delinquency as a prediction problem and scans every loan, every day.",
      "Instead of static scorecards, we deliver a living risk signal that updates with market conditions and property performance so surprises are surfaced early.",
    ],
  },
  {
    title: "Data: multifamily loans with outcomes",
    body: [
      "We train on tens of thousands of multifamily loans with true 60+ day delinquency outcomes across vintages, geographies, and property types.",
      "Features span loan structure (DSCR, LTV, rate, term), property (units, age, location), and performance signals. Inputs stay familiar to underwriters while the model learns from reality.",
      "Data is cleaned, standardized, and aligned to lender-available fields, keeping adoption low-friction while preserving signal quality.",
    ],
  },
  {
    title: "Modeling: from baseline to boosting",
    body: [
      "We started with a transparent logistic regression baseline to benchmark separability between delinquent and clean loans.",
      "To capture interactions—how leverage, market softness, and vintage combine—we advanced to gradient-boosted trees (XGBoost) that learn nonlinear patterns a linear model misses.",
      "The ensemble structure balances flexibility and control, letting us tune depth, learning rate, and regularization for stability across vintages.",
    ],
  },
  {
    title: "Selecting the optimal strategy",
    body: [
      "We evaluated models on out-of-sample data with risk-team metrics: recall on delinquents and precision on alerts.",
      "Logistic regression delivered recall but flooded teams with false positives. XGBoost balanced recall and precision at practical thresholds, making alerts actionable.",
      "Thresholds are calibrated to maximize actionable alerts per analyst hour, reducing noise while keeping early warnings intact.",
    ],
  },
  {
    title: "What Praedium delivers for lenders",
    body: [
      "Praedium surfaces forward-looking delinquency risk so teams can prioritize outreach, workouts, and portfolio stress tests before loans hit 60+ days past due.",
      "Because inputs mirror existing workflows—DSCR, LTV, structure, property performance—the tool slots into credit and asset management while continuously reweighting signals as markets move.",
      "Output is presented as clear probabilities and bands with contextual distributions, making it easy to brief committees and align actions with modeled risk.",
    ],
  },
];

export const StorySection = ({
  stats,
  inputs,
  loading,
  summaryError,
}: StorySectionProps) => (
  <section className="story-section" id="story">
    <div className="thesis-grid">
      {thesisSections.map((section, idx) => {
        const isEven = idx % 2 === 0;
        return (
          <article
            key={section.title}
            className={clsx(
              "thesis-section",
              isEven ? "thesis-purple" : "thesis-silver",
              isEven ? "align-left" : "align-right"
            )}
          >
            <p className="eyebrow thesis-eyebrow">{section.title}</p>
            {section.body.map((paragraph) => (
              <p key={paragraph}>{paragraph}</p>
            ))}
          </article>
        );
      })}
    </div>
    <div className="story-panel stats">
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
    </div>
    <div className="story-panel full thesis-cta">
      <p className="eyebrow">Ready when you are</p>
      <h3>Use the predictive experience now.</h3>
      <PredictivePanel
        variant="inline"
        inputs={inputs}
        loading={loading}
        summaryError={summaryError}
      />
    </div>
  </section>
);
