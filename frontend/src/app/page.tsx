"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import dynamic from "next/dynamic";
import { getScores, Score } from "@/lib/api";
import ScoreCard from "@/components/scores/ScoreCard";
import { fmt } from "@/lib/utils";

const MapView = dynamic(() => import("@/components/map/MapView"), { ssr: false });

export default function Dashboard() {
  const router = useRouter();
  const [scores, setScores] = useState<Score[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [stateFilter, setStateFilter] = useState("");

  useEffect(() => {
    getScores({ limit: "100", state: stateFilter })
      .then((res) => {
        setScores(res.data);
        setError(null);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [stateFilter]);

  const handleMapSelect = useCallback(
    (fips: string) => router.push(`/location/${fips}`),
    [router]
  );

  const avgPsi = scores.length ? scores.reduce((s, r) => s + r.psi, 0) / scores.length : 0;
  const highCount = scores.filter((s) => s.psi >= 70).length;
  const lowCount = scores.filter((s) => s.psi < 50).length;

  return (
    <div className="mx-auto max-w-7xl px-6 py-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold">Dashboard</h1>
        <p className="text-sm text-[var(--muted)]">
          Predictive Suitability Index for U.S. data center locations
        </p>
      </div>

      <div className="mb-6 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <StatCard label="Counties Scored" value={scores.length.toString()} />
        <StatCard label="Avg PSI" value={fmt(avgPsi)} />
        <StatCard label="High Suitability" value={highCount.toString()} accent="text-green-400" />
        <StatCard label="Low Suitability" value={lowCount.toString()} accent="text-red-400" />
      </div>

      {error && (
        <div className="mb-4 rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-400">
          API Error: {error}. Make sure the FastAPI backend is running on port 8000.
        </div>
      )}

      <div className="grid gap-6 lg:grid-cols-[1fr_380px]">
        <div className="h-[500px] rounded-lg border border-[var(--card-border)] overflow-hidden">
          {loading ? (
            <div className="flex h-full items-center justify-center text-[var(--muted)]">
              Loading map data...
            </div>
          ) : (
            <MapView scores={scores} onSelect={handleMapSelect} />
          )}
        </div>

        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold text-[var(--muted)]">TOP LOCATIONS</h2>
            <select
              value={stateFilter}
              onChange={(e) => setStateFilter(e.target.value)}
              className="rounded-md border border-[var(--card-border)] bg-[var(--card)] px-2 py-1 text-xs text-[var(--foreground)]"
            >
              <option value="">All States</option>
              {Array.from(new Set(scores.map((s) => s.state).filter(Boolean))).sort().map((st) => (
                <option key={st} value={st}>{st}</option>
              ))}
            </select>
          </div>

          <div className="max-h-[460px] space-y-2 overflow-y-auto pr-1">
            {scores.slice(0, 15).map((s, i) => (
              <ScoreCard key={s.county_fips} score={s} rank={i + 1} />
            ))}
            {!loading && scores.length === 0 && (
              <div className="py-8 text-center text-sm text-[var(--muted)]">
                No scores available
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function StatCard({ label, value, accent }: { label: string; value: string; accent?: string }) {
  return (
    <div className="rounded-lg border border-[var(--card-border)] bg-[var(--card)] p-3">
      <div className={`text-xl font-bold font-mono ${accent || ""}`}>{value}</div>
      <div className="text-xs text-[var(--muted)]">{label}</div>
    </div>
  );
}
