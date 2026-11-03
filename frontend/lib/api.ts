export type ListingData = {
  source: string;
  source_id: string;
  title: string;
  brand: string | null;
  model: string | null;
  category: string | null;
  attributes: Record<string, string | null>;
  price: string | null;
  currency: string | null;
  raw?: Record<string, unknown>;
};
export type Listing = { id: string; revision: number; data: ListingData };
export type Identity = {
  id: string;
  level: string;
  parent_id: string | null;
  members: string[];
};
export type Candidate = {
  listing_id: string;
  rank: number;
  sources: string[];
  text_score: number;
};
export type Pair = {
  id: string;
  left: string;
  right: string;
  left_revision: number;
  right_revision: number;
  probability: number | null;
  model_decision: string;
  needs_review: boolean;
  hierarchy: {
    family: string;
    variant: string;
    relation: string;
    conflicts: string[];
    missing: string[];
  };
  features: Record<string, number>;
};
export type GraphEdge = {
  left: string;
  right: string;
  level: string;
  score: number;
  reason?: string;
};
export type Overview = {
  catalog_revision: number;
  identities_stale: boolean;
  model_loaded: boolean;
  graph: { accepted?: GraphEdge[]; rejected?: GraphEdge[] };
  history: {
    event: string;
    level: string;
    revision: number;
    previous_id?: string;
    identity_id?: string;
  }[];
};
export type Job = {
  id: string;
  kind: string;
  status: string;
  error: string | null;
};
export type Benchmark = {
  name: string;
  phase: string;
  precision: number | null;
  recall: number | null;
  average_precision: number;
  retrieval: Record<string, number>;
  note: string;
};
export type DevelopmentModel = {
  name: string;
  precision: number | null;
  recall: number | null;
  true_positives: number;
  false_positives: number;
  average_precision: number;
};
export type DevelopmentBenchmark = {
  protocol: string;
  warning: string;
  pairs: number;
  models: DevelopmentModel[];
};
export type EvaluationReport = {
  id: string;
  data: {
    phase?: string;
    dataset: string;
    synthetic: boolean;
    feature_version?: string;
    models?: Record<
      string,
      {
        classification: Omit<DevelopmentModel, "name" | "average_precision">;
        average_precision: number;
      }
    >;
  };
};
export type RetrievalCostReport = {
  synthetic: boolean;
  scope: string;
  runs: {
    listings: number;
    query_p50_ms: number;
    query_p95_ms: number;
    build_seconds_including_vocabulary: number;
    queries_per_second: number;
    sampled_peak_rss_bytes: number;
  }[];
};

export async function api<T>(path: string, payload?: unknown): Promise<T> {
  const response = await fetch(`/api/catalog${path}`, {
    method: payload === undefined ? "GET" : "POST",
    headers: { "Content-Type": "application/json" },
    body: payload === undefined ? undefined : JSON.stringify(payload),
  });
  const result = await response.json();
  if (!response.ok)
    throw new Error(
      typeof result.detail === "string"
        ? result.detail
        : "The request could not be completed. Check the supplied fields.",
    );
  return result;
}

export const shortId = (id: string) => id.slice(0, 8);
export function priceLabel(listing: Listing) {
  const { price, currency } = listing.data;
  if (!price) return "—";
  const amount = Number(price);
  if (!Number.isFinite(amount) || !currency) return price;
  try {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency,
      maximumFractionDigits: 2,
    }).format(amount);
  } catch {
    return `${price} ${currency || ""}`;
  }
}
export const relationLabel: Record<string, string> = {
  same_exact_variant: "Same exact variant",
  same_family_different_variant: "Same family · different variant",
  different_product: "Different product",
  uncertain: "Needs review",
};
