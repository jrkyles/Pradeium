export type InputFieldMeta = {
  key: string;
  label: string;
  type: "numeric" | "categorical";
  min?: number;
  max?: number;
  median?: number;
  options?: string[];
};

export type SummaryResponse = {
  message: string;
  stats: {
    medianLTV: string;
    medianDSCR: string;
    medianLoanSize: string;
    medianNoteRate: string;
    medianPropertyAge: string;
  };
  inputs: InputFieldMeta[];
  features: string[];
};

export type ProbabilityPoint = {
  category: string;
  probability: number;
};

export type PredictionResponse = {
  rating: string;
  probability: number;
  distribution: ProbabilityPoint[];
  chart: string;
};

export type CorrelationResponse = {
  chart: string;
};

