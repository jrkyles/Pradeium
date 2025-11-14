import { useEffect, useMemo, useState } from "react";
import { HeroSection } from "../components/HeroSection";
import { StorySection } from "../components/StorySection";
import { fetchSummary } from "../api";
import type { InputFieldMeta, SummaryResponse } from "../types/api";
import { BrandHeader } from "../components/BrandHeader";

export const LandingPage = () => {
  const [summary, setSummary] = useState<SummaryResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [scrollProgress, setScrollProgress] = useState(0);

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

  useEffect(() => {
    const updateScroll = () => {
      const progress = Math.min(
        window.scrollY / Math.max(window.innerHeight, 1),
        1
      );
      setScrollProgress(progress);
    };
    updateScroll();
    window.addEventListener("scroll", updateScroll, { passive: true });
    return () => window.removeEventListener("scroll", updateScroll);
  }, []);

  const inputs: InputFieldMeta[] = useMemo(
    () => summary?.inputs ?? [],
    [summary]
  );
  const stats =
    summary?.stats ?? {
      medianNOI: "—",
      medianInterestCoverage: "—",
      medianLeverage: "—",
      medianDefaultRate: "—",
    };

  return (
    <>
      <BrandHeader isVisible={scrollProgress > 0.25} />
      <main className="landing-layout">
        <HeroSection
          inputs={inputs}
          loading={loading}
          summaryError={error}
          fadeProgress={scrollProgress}
        />
        <StorySection
          stats={stats}
          inputs={inputs}
          loading={loading}
          summaryError={error}
        />
      </main>
    </>
  );
};
