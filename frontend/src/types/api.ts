export type InputFieldMeta = {
  key: string;
  label: string;
  min: number;
  max: number;
  median: number;
};

export type SummaryResponse = {
  message: string;
  stats: {
    medianNOI: string;
    medianInterestCoverage: string;
    medianLeverage: string;
    medianDefaultRate: string;
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


