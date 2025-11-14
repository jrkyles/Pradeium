import { useEffect, useState } from "react";
import { HeroSection } from "../components/HeroSection";
import { StorySection } from "../components/StorySection";
import { DemoForm } from "../components/DemoForm";
import { fetchSummary } from "../api";
import type { InputFieldMeta, SummaryResponse } from "../types/api";

export const LandingPage = () => {
  const [summary, setSummary] = useState<SummaryResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showForm, setShowForm] = useState(false);

  useEffect(() => {
    const loadSummary = async () => {
      try {
        const data = await fetchSummary();
        setSummary(data);
      } catch {
        setError("We couldn’t load the Praedium summary.");
      } finally {
        setLoading(false);
      }
    };
    loadSummary();
  }, []);

  const handleDemoClick = () => {
    setShowForm(true);
    const formElement = document.querySelector(".demo-form");
    formElement?.scrollIntoView({ behavior: "smooth" });
  };

  const inputs: InputFieldMeta[] = summary?.inputs ?? [];
  const stats =
    summary?.stats ?? {
      medianNOI: "—",
      medianInterestCoverage: "—",
      medianLeverage: "—",
      medianDefaultRate: "—",
    };

  return (
    <main>
      <HeroSection onDemoClick={handleDemoClick} />
      <div className="transition-panel" />
      <StorySection stats={stats} onDemoClick={handleDemoClick} />
      <DemoForm
        inputs={inputs}
        isVisible={showForm}
        loading={loading}
        summaryError={error}
      />
    </main>
  );
};

