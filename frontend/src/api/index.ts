import { apiClient } from "./client";
import type {
  CorrelationResponse,
  PredictionResponse,
  SummaryResponse,
} from "../types/api";

export const fetchSummary = async (): Promise<SummaryResponse> => {
  const { data } = await apiClient.get<SummaryResponse>("/api/summary");
  return data;
};

export const fetchPrediction = async (
  payload: Record<string, number | string>
): Promise<PredictionResponse> => {
  const { data } = await apiClient.post<PredictionResponse>(
    "/api/predict",
    payload
  );
  return data;
};

export const fetchCorrelations = async (): Promise<CorrelationResponse> => {
  const { data } = await apiClient.get<CorrelationResponse>("/api/correlations");
  return data;
};


