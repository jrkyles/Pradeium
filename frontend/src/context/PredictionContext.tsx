import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import type { ReactNode } from "react";
import type { PredictionResponse } from "../types/api";

type PredictionContextValue = {
  prediction: PredictionResponse | null;
  setPrediction: (prediction: PredictionResponse | null) => void;
};

const PredictionContext = createContext<PredictionContextValue | undefined>(
  undefined
);

export const PredictionProvider = ({ children }: { children: ReactNode }) => {
  const [prediction, setPredictionState] = useState<PredictionResponse | null>(
    null
  );

  const setPrediction = useCallback((value: PredictionResponse | null) => {
    setPredictionState(value);
    if (value) {
      sessionStorage.setItem("praedium.prediction", JSON.stringify(value));
    } else {
      sessionStorage.removeItem("praedium.prediction");
    }
  }, []);

  useEffect(() => {
    const stored = sessionStorage.getItem("praedium.prediction");
    if (stored && !prediction) {
      try {
        setPredictionState(JSON.parse(stored));
      } catch {
        sessionStorage.removeItem("praedium.prediction");
      }
    }
  }, [prediction]);

  const value = useMemo(
    () => ({
      prediction,
      setPrediction,
    }),
    [prediction, setPrediction]
  );

  return (
    <PredictionContext.Provider value={value}>
      {children}
    </PredictionContext.Provider>
  );
};

export const usePrediction = (): PredictionContextValue => {
  const context = useContext(PredictionContext);
  if (!context) {
    throw new Error("usePrediction must be used within a PredictionProvider");
  }
  return context;
};

