"use client";

import Link from "next/link";
import { Score } from "@/lib/api";
import { psiColor, fipsToState } from "@/lib/utils";

export default function ScoreCard({ score, rank }: { score: Score; rank: number }) {
  const color = psiColor(score.psi);

  return (
    <Link
      href={`/location/${score.county_fips}`}
      className="flex items-center gap-4 rounded-lg border border-[var(--card-border)] bg-[var(--card)] p-3 transition-colors hover:border-[var(--accent)]/40"
    >
      <div
        className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-full text-sm font-bold"
        style={{ backgroundColor: `${color}20`, color }}
      >
        {rank}
      </div>

      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className="truncate text-sm font-medium">
            {score.county_name || score.county_fips}
          </span>
          {score.state && (
            <span className="text-xs text-[var(--muted)]">
              {score.state || fipsToState(score.county_fips)}
            </span>
          )}
        </div>
        <div className="mt-1 flex gap-3 text-xs text-[var(--muted)]">
          <span>PWR {score.power_score?.toFixed(0) ?? "—"}</span>
          <span>ENV {score.environmental_score?.toFixed(0) ?? "—"}</span>
          <span>SOC {score.social_score?.toFixed(0) ?? "—"}</span>
        </div>
      </div>

      <div className="text-right">
        <div className="text-lg font-bold font-mono" style={{ color }}>
          {score.psi?.toFixed(1)}
        </div>
        <div className="text-[10px] text-[var(--muted)]">PSI</div>
      </div>
    </Link>
  );
}
