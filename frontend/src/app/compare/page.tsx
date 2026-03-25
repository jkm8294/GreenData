"use client";

import { useEffect, useState } from "react";
import { getScores, getCompare, Score } from "@/lib/api";
import RadarOverlay from "@/components/compare/RadarOverlay";
import DimensionBars from "@/components/scores/DimensionBars";
import { psiColor } from "@/lib/utils";

export default function ComparePage() {
  const [allScores, setAllScores] = useState<Score[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [compared, setCompared] = useState<Score[]>([]);
  const [loading, setLoading] = useState(false);

  // Load all scores for the picker
  useEffect(() => {
    getScores({ limit: "200" }).then((res) => setAllScores(res.data));
  }, []);

  // Fetch comparison data when selection changes
  useEffect(() => {
    if (selected.length < 2) {
      setCompared([]);
      return;
    }
    setLoading(true);
    getCompare(selected)
      .then((res) => setCompared(res.data))
      .catch(() => setCompared([]))
      .finally(() => setLoading(false));
  }, [selected]);

  const toggleSelect = (fips: string) => {
    setSelected((prev) =>
      prev.includes(fips)
        ? prev.filter((f) => f !== fips)
        : prev.length < 5
          ? [...prev, fips]
          : prev
    );
  };

  return (
    <div className="mx-auto max-w-7xl px-6 py-6">
      <h1 className="mb-1 text-2xl font-bold">Compare Locations</h1>
      <p className="mb-6 text-sm text-[var(--muted)]">
        Select 2–5 counties to compare side by side
      </p>

      <div className="grid gap-6 lg:grid-cols-[320px_1fr]">
        {/* Picker */}
        <div className="space-y-2">
          <h2 className="text-xs font-semibold text-[var(--muted)]">
            SELECT LOCATIONS ({selected.length}/5)
          </h2>
          <div className="max-h-[600px] space-y-1 overflow-y-auto rounded-lg border border-[var(--card-border)] bg-[var(--card)] p-2">
            {allScores.map((s) => {
              const isSelected = selected.includes(s.county_fips);
              return (
                <button
                  key={s.county_fips}
                  onClick={() => toggleSelect(s.county_fips)}
                  className={`flex w-full items-center justify-between rounded-md px-3 py-2 text-left text-sm transition-colors ${
                    isSelected
                      ? "bg-[var(--accent)]/10 text-[var(--accent)]"
                      : "hover:bg-[var(--card-border)]/50"
                  }`}
                >
                  <span className="truncate">
                    {s.county_name || s.county_fips}
                    {s.state && (
                      <span className="ml-1 text-xs text-[var(--muted)]">{s.state}</span>
                    )}
                  </span>
                  <span className="ml-2 font-mono text-xs" style={{ color: psiColor(s.psi) }}>
                    {s.psi.toFixed(1)}
                  </span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Results */}
        <div>
          {selected.length < 2 ? (
            <div className="flex h-96 items-center justify-center rounded-lg border border-dashed border-[var(--card-border)] text-[var(--muted)]">
              Select at least 2 locations to compare
            </div>
          ) : loading ? (
            <div className="flex h-96 items-center justify-center text-[var(--muted)]">
              Loading comparison...
            </div>
          ) : (
            <div className="space-y-6">
              {/* Radar chart */}
              <div className="rounded-lg border border-[var(--card-border)] bg-[var(--card)] p-6">
                <h2 className="mb-2 text-sm font-semibold text-[var(--muted)]">
                  RADAR COMPARISON
                </h2>
                <RadarOverlay locations={compared} />
              </div>

              {/* Side by side cards */}
              <div className="grid gap-4" style={{ gridTemplateColumns: `repeat(${Math.min(compared.length, 3)}, 1fr)` }}>
                {compared.map((loc) => (
                  <div
                    key={loc.county_fips}
                    className="rounded-lg border border-[var(--card-border)] bg-[var(--card)] p-5"
                  >
                    <div className="mb-1 text-sm font-semibold">
                      {loc.county_name || loc.county_fips}
                    </div>
                    <div className="mb-4 text-xs text-[var(--muted)]">
                      {loc.state} &middot; FIPS {loc.county_fips}
                    </div>

                    <div className="mb-4 text-center">
                      <span
                        className="text-3xl font-bold font-mono"
                        style={{ color: psiColor(loc.psi) }}
                      >
                        {loc.psi.toFixed(1)}
                      </span>
                      <span className="ml-1 text-xs text-[var(--muted)]">PSI</span>
                    </div>

                    <DimensionBars
                      power={loc.power_score ?? 0}
                      environmental={loc.environmental_score ?? 0}
                      social={loc.social_score ?? 0}
                    />

                    <div className="mt-3 text-right text-xs text-[var(--muted)]">
                      Confidence: {loc.confidence != null ? `${(loc.confidence * 100).toFixed(0)}%` : "—"}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
