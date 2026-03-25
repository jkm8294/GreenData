const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function fetchAPI<T>(path: string, params?: Record<string, string>): Promise<T> {
  const url = new URL(`${API_BASE}${path}`);
  if (params) {
    Object.entries(params).forEach(([k, v]) => {
      if (v !== undefined && v !== null && v !== "") url.searchParams.set(k, v);
    });
  }
  const res = await fetch(url.toString(), { next: { revalidate: 60 } });
  if (!res.ok) throw new Error(`API ${res.status}: ${res.statusText}`);
  return res.json();
}

/* ---------- Types ---------- */

export interface Score {
  county_fips: string;
  state?: string;
  county_name?: string;
  lat?: number;
  lon?: number;
  psi: number;
  power_score: number;
  environmental_score: number;
  social_score: number;
  confidence: number;
  rank?: number;
  model_version?: string;
  // optional enrichments
  raw_features?: Record<string, number | null>;
  resistance_detail?: Record<string, number | null>;
}

export interface ValidationData {
  model_version: string | null;
  auc: number | null;
  macro_f1: number | null;
  social_test_p_value: number | null;
  n_validation_sites: number | null;
  class_balance: { successful: number | null; blocked: number | null } | null;
  imbalance_strategy: string | null;
  geocoding_match_rate: number | null;
  weights: Record<string, number> | null;
  sensitivity: { scenario_1: string; scenario_2: string; kendall_tau: number; p_value: number; stability: string }[] | null;
}

export interface Topic {
  topic_id: number;
  label: string;
  top_words: string;
}

/* ---------- Endpoints ---------- */

export async function getScores(params?: {
  min_psi?: string;
  state?: string;
  limit?: string;
  sort?: string;
  order?: string;
}): Promise<{ data: Score[]; count: number; total: number }> {
  return fetchAPI("/api/scores", params);
}

export async function getScoreDetail(fips: string): Promise<{ data: Score }> {
  return fetchAPI(`/api/scores/${fips}`);
}

export async function getValidation(): Promise<ValidationData> {
  return fetchAPI("/api/validation");
}

export async function getTopics(): Promise<{ data: Topic[]; count: number }> {
  return fetchAPI("/api/topics");
}

export async function getStates(): Promise<{ data: string[] }> {
  return fetchAPI("/api/states");
}

export async function getCompare(fipsList: string[]): Promise<{ data: Score[]; count: number }> {
  const url = new URL(`${API_BASE}/api/compare`);
  fipsList.forEach((f) => url.searchParams.append("fips", f));
  const res = await fetch(url.toString(), { next: { revalidate: 60 } });
  if (!res.ok) throw new Error(`API ${res.status}`);
  return res.json();
}

export async function getHealth(): Promise<Record<string, unknown>> {
  return fetchAPI("/api/health");
}
