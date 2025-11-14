type StorySectionProps = {
  stats: {
    medianNOI: string;
    medianInterestCoverage: string;
    medianLeverage: string;
    medianDefaultRate: string;
  };
  onDemoClick: () => void;
};

export const StorySection = ({ stats, onDemoClick }: StorySectionProps) => (
  <section className="story-section">
    <div className="story-grid">
      <article>
        <h2>Our story</h2>
        <p>
          Built at USC, Praedium pairs institutional research with pragmatic data
          engineering to give sponsors and lenders a clear look into the credit
          health of income-producing assets.
        </p>
      </article>
      <article>
        <h3>The data</h3>
        <p>
          Cleaned, normalized, and stress-tested across {stats.medianDefaultRate} average
          five-year default risk, our dataset captures the levers that move credit
          quality in CRE portfolios.
        </p>
      </article>
      <article>
        <h3>The model</h3>
        <p>
          The Praedium logistic regression engine calibrates against market
          benchmarks, focusing on key drivers like NOI, leverage, and coverage
          to surface a defensible letter-grade prediction.
        </p>
      </article>
      <article className="stat-card">
        <p>Median NOI</p>
        <strong>{stats.medianNOI}</strong>
        <p>Coverage ratio</p>
        <strong>{stats.medianInterestCoverage}x</strong>
        <p>Net leverage</p>
        <strong>{stats.medianLeverage}x</strong>
      </article>
    </div>
    <button className="primary-btn secondary" onClick={onDemoClick}>
      Demo the tool
    </button>
  </section>
);

