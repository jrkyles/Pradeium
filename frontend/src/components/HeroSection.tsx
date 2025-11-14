type HeroSectionProps = {
  onDemoClick: () => void;
};

export const HeroSection = ({ onDemoClick }: HeroSectionProps) => (
  <section className="hero-section">
    <div className="hero-content">
      <p className="hero-kicker">Praedium</p>
      <h1>Institutional-grade credit intelligence for real assets.</h1>
      <p>
        Translate operating data into an actionable credit view powered by
        modern machine learning and the rigor of commercial real estate
        underwriting.
      </p>
      <button className="primary-btn" onClick={onDemoClick}>
        Demo the tool
      </button>
    </div>
  </section>
);

