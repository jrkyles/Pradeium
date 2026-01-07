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
  lastPrediction: PredictionResponse | null;
  setPrediction: (prediction: PredictionResponse | null) => void;
};

const PredictionContext = createContext<PredictionContextValue | undefined>(
  undefined
);

export const PredictionProvider = ({ children }: { children: ReactNode }) => {
  const [prediction, setPredictionState] = useState<PredictionResponse | null>(
    null
  );
  const [lastPrediction, setLastPrediction] =
    useState<PredictionResponse | null>(null);

  const setPrediction = useCallback((value: PredictionResponse | null) => {
    setPredictionState((current) => {
      if (value && current) {
        setLastPrediction(current);
        sessionStorage.setItem("praedium.prediction.last", JSON.stringify(current));
      }
      return value;
    });
    if (value) {
      sessionStorage.setItem("praedium.prediction.current", JSON.stringify(value));
    } else {
      sessionStorage.removeItem("praedium.prediction.current");
      sessionStorage.removeItem("praedium.prediction.last");
      setLastPrediction(null);
    }
  }, []);

  useEffect(() => {
    const storedCurrent = sessionStorage.getItem("praedium.prediction.current");
    const storedLast = sessionStorage.getItem("praedium.prediction.last");
    if (storedCurrent && !prediction) {
      try {
        setPredictionState(JSON.parse(storedCurrent));
      } catch {
        sessionStorage.removeItem("praedium.prediction.current");
      }
    }
    if (storedLast && !lastPrediction) {
      try {
        setLastPrediction(JSON.parse(storedLast));
      } catch {
        sessionStorage.removeItem("praedium.prediction.last");
      }
    }
  }, [prediction, lastPrediction]);

  const value = useMemo(
    () => ({
      prediction,
      lastPrediction,
      setPrediction,
    }),
    [prediction, lastPrediction, setPrediction]
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
