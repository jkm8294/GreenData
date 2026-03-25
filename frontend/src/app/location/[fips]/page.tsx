"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { getScoreDetail, Score } from "@/lib/api";
import PSIGauge from "@/components/scores/PSIGauge";
import DimensionBars from "@/components/scores/DimensionBars";
import { fmt, fipsToState } from "@/lib/utils";

export default function LocationPage() {
  const { fips } = useParams<{ fips: string }>();
  const [data, setData] = useState<Score | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!fips) return;
    getScoreDetail(fips)
      .then((res) => setData(res.data))
      .catch((e) => setError(e.message));
  }, [fips]);

  if (error) {
    return (
      <div className="mx-auto max-w-4xl px-6 py-12 text-center">
        <h1 className="text-xl font-bold text-red-400">County not found</h1>
        <p className="mt-2 text-sm text-[var(--muted)]">
          No PSI score for FIPS {fips}. {error}
        </p>
        <Link href="/" className="mt-4 inline-block text-sm text-[var(--accent)] hover:underline">
          Back to Dashboard
        </Link>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="flex min-h-[60vh] items-center justify-center text-[var(--muted)]">
        Loading...
      </div>
    );
  }

  const resistance = data.resistance_detail;
  const features = data.raw_features;

  return (
    <div className="mx-auto max-w-5xl px-6 py-6">
      {/* Breadcrumb */}
      <div className="mb-6 text-sm text-[var(--muted)]">
        <Link href="/" className="hover:text-[var(--foreground)]">Dashboard</Link>
        <span className="mx-2">/</span>
        <span className="text-[var(--foreground)]">
          {data.county_name || fips}
        </span>
      </div>

      {/* Header */}
      <div className="mb-8 flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold">{data.county_name || `FIPS ${fips}`}</h1>
          <p className="text-[var(--muted)]">
            {data.state || fipsToState(fips)} &middot; FIPS {fips}
            {data.model_version && (
              <span className="ml-2 text-xs">Model v{data.model_version}</span>
            )}
          </p>
        </div>
        <div className="text-right">
          <div className="text-sm text-[var(--muted)]">Confidence</div>
          <div className="font-mono text-lg font-semibold">
            {data.confidence != null ? `${(data.confidence * 100).toFixed(0)}%` : "—"}
          </div>
        </div>
      </div>

      <div className="grid gap-6 md:grid-cols-[220px_1fr]">
        {/* PSI Gauge */}
        <div className="flex flex-col items-center gap-4 rounded-lg border border-[var(--card-border)] bg-[var(--card)] p-6">
          <PSIGauge score={data.psi} />
          {data.rank && (
            <div className="text-center text-xs text-[var(--muted)]">
              Ranked <span className="font-semibold text-[var(--foreground)]">#{data.rank}</span> overall
            </div>
          )}
        </div>

        {/* Dimension breakdown */}
        <div className="rounded-lg border border-[var(--card-border)] bg-[var(--card)] p-6">
          <h2 className="mb-4 text-sm font-semibold text-[var(--muted)]">DIMENSION SCORES</h2>
          <DimensionBars
            power={data.power_score ?? 0}
            environmental={data.environmental_score ?? 0}
            social={data.social_score ?? 0}
          />
        </div>
      </div>

      {/* Detail sections */}
      <div className="mt-6 grid gap-6 md:grid-cols-2">
        {/* Raw features */}
        {features && (
          <div className="rounded-lg border border-[var(--card-border)] bg-[var(--card)] p-6">
            <h2 className="mb-4 text-sm font-semibold text-[var(--muted)]">RAW FEATURES</h2>
            <div className="space-y-2 text-sm">
              {Object.entries(features)
                .filter(([k]) => !["county_fips", "state_fips", "state"].includes(k))
                .map(([key, val]) => (
                  <div key={key} className="flex justify-between">
                    <span className="text-[var(--muted)]">{key.replace(/_/g, " ")}</span>
                    <span className="font-mono">{val != null ? (typeof val === "number" ? fmt(val, 2) : String(val)) : "—"}</span>
                  </div>
                ))}
            </div>
          </div>
        )}

        {/* Social resistance detail */}
        {resistance && (
          <div className="rounded-lg border border-[var(--card-border)] bg-[var(--card)] p-6">
            <h2 className="mb-4 text-sm font-semibold text-[var(--muted)]">SOCIAL RESISTANCE</h2>
            <div className="space-y-2 text-sm">
              <Row label="Reddit Posts" value={resistance.post_count} />
              <Row label="Avg Sentiment" value={resistance.avg_sentiment} signed />
              <Row label="Negative Post %" value={resistance.negative_pct} suffix="%" />
              <Row label="Posts per Capita" value={resistance.posts_per_capita} decimals={6} />
              <Row label="Intensity Factor" value={resistance.intensity_factor} />
              <Row label="Volume Factor" value={resistance.volume_factor} />
              <div className="border-t border-[var(--card-border)] pt-2">
                <Row label="Resistance Score (raw)" value={resistance.raw_resistance_mean} />
                <Row label="Resistance Score (corrected)" value={resistance.social_resistance_score} bold />
              </div>
            </div>

            {/* Vocal minority note */}
            {resistance.population != null && (
              <div className="mt-4 rounded-md bg-amber-500/10 px-3 py-2 text-xs text-amber-400">
                <strong>Vocal Minority Correction: </strong>
                {resistance.post_count ?? 0} posts in a county of{" "}
                {Number(resistance.population).toLocaleString()} people.
                The intensity factor adjusts for population size to prevent
                large counties from dominating the resistance signal.
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function Row({
  label,
  value,
  decimals = 3,
  signed,
  suffix = "",
  bold,
}: {
  label: string;
  value: number | null | undefined;
  decimals?: number;
  signed?: boolean;
  suffix?: string;
  bold?: boolean;
}) {
  const display =
    value != null
      ? `${signed && value > 0 ? "+" : ""}${value.toFixed(decimals)}${suffix}`
      : "—";
  return (
    <div className="flex justify-between">
      <span className="text-[var(--muted)]">{label}</span>
      <span className={`font-mono ${bold ? "font-semibold text-[var(--foreground)]" : ""}`}>
        {display}
      </span>
    </div>
  );
}
