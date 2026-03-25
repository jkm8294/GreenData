"use client";

import { useEffect, useState } from "react";
import { getValidation, getTopics, ValidationData, Topic } from "@/lib/api";
import WeightComparison from "@/components/validation/WeightComparison";
import ClassBalance from "@/components/validation/ClassBalance";

export default function MethodologyPage() {
  const [validation, setValidation] = useState<ValidationData | null>(null);
  const [topics, setTopics] = useState<Topic[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getValidation()
      .then(setValidation)
      .catch((e) => setError(e.message));
    getTopics()
      .then((res) => setTopics(res.data))
      .catch(() => {});
  }, []);

  if (error) {
    return (
      <div className="mx-auto max-w-4xl px-6 py-12 text-center text-red-400">
        Failed to load validation data: {error}
      </div>
    );
  }

  if (!validation) {
    return (
      <div className="flex min-h-[60vh] items-center justify-center text-[var(--muted)]">
        Loading methodology data...
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-4xl px-6 py-6">
      <div className="mb-8">
        <h1 className="text-2xl font-bold">Methodology</h1>
        <p className="text-sm text-[var(--muted)]">
          Complete analytical proof — data pipeline, NLP, feature engineering, validation.
          {validation.model_version && (
            <span className="ml-2 rounded bg-[var(--card)] px-2 py-0.5 text-xs font-mono border border-[var(--card-border)]">
              Model v{validation.model_version}
            </span>
          )}
        </p>
      </div>

      <div className="space-y-8">
        {/* ── Section 1: Data Pipeline ── */}
        <Section number={1} title="Data Pipeline">
          <p className="mb-4 text-sm text-[var(--muted)]">
            Government data from four federal APIs merged into a single feature matrix
            keyed by county FIPS code. Social signal from a frozen Reddit corpus
            scraped via Firecrawl.
          </p>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <MetricCard label="Data Sources" value="5" sub="EIA, NREL, NOAA, USGS, Reddit" />
            <MetricCard label="Target Counties" value="26" sub="Across 15 states" />
            <MetricCard
              label="Geocoding Match Rate"
              value={validation.geocoding_match_rate ? `${(validation.geocoding_match_rate * 100).toFixed(0)}%` : "—"}
              sub="Reddit posts → county FIPS"
            />
            <MetricCard
              label="Validation Sites"
              value={validation.n_validation_sites?.toString() || "—"}
              sub="Successful + blocked projects"
            />
          </div>

          <div className="mt-4 rounded-lg border border-[var(--card-border)] bg-[var(--card)] p-4 text-xs text-[var(--muted)] space-y-1">
            <div><strong className="text-[var(--foreground)]">EIA Open Data API v2</strong> — Industrial electricity prices, generation by fuel type, renewable percentage</div>
            <div><strong className="text-[var(--foreground)]">NREL PVWatts + Wind Toolkit</strong> — Solar GHI (kWh/m²/day), wind capacity factor per county centroid</div>
            <div><strong className="text-[var(--foreground)]">NOAA CDO Web Services</strong> — Average temperature, cooling degree days, extreme weather events</div>
            <div><strong className="text-[var(--foreground)]">USGS NWIS</strong> — Groundwater depth, surface water discharge (streamflow)</div>
            <div><strong className="text-[var(--foreground)]">Reddit via Firecrawl</strong> — Frozen corpus of data center community discourse (one-time scrape)</div>
          </div>
        </Section>

        {/* ── Section 2: NLP Analysis ── */}
        <Section number={2} title="NLP Analysis">
          <p className="mb-4 text-sm text-[var(--muted)]">
            LDA topic modeling discovers <em>what</em> people oppose, not just whether
            they&apos;re negative. VADER sentiment with domain-specific lexicon scores each
            post. Vocal minority correction prevents large counties from dominating.
          </p>

          {/* Topics */}
          {topics.length > 0 && (
            <div className="mb-4">
              <h3 className="mb-2 text-xs font-semibold text-[var(--muted)]">LDA TOPICS (6 clusters)</h3>
              <div className="grid gap-2 sm:grid-cols-2">
                {topics.map((t) => (
                  <div key={t.topic_id} className="rounded-lg border border-[var(--card-border)] bg-[var(--card)] p-3">
                    <div className="flex items-center gap-2">
                      <span className="flex h-5 w-5 items-center justify-center rounded bg-[var(--accent)]/10 text-[10px] font-bold text-[var(--accent)]">
                        {t.topic_id}
                      </span>
                      <span className="text-sm font-medium">{t.label}</span>
                    </div>
                    <p className="mt-1 text-xs text-[var(--muted)] leading-relaxed">
                      {typeof t.top_words === "string"
                        ? t.top_words.replace(/[\[\]']/g, "")
                        : Array.isArray(t.top_words) ? (t.top_words as string[]).join(", ") : ""}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Domain lexicon */}
          <div className="rounded-lg border border-[var(--card-border)] bg-[var(--card)] p-4">
            <h3 className="mb-2 text-xs font-semibold text-[var(--muted)]">DOMAIN LEXICON ADDITIONS</h3>
            <div className="grid grid-cols-2 gap-x-6 gap-y-1 text-xs sm:grid-cols-3">
              {[
                ["moratorium", -2.5], ["water shortage", -3.0], ["noise pollution", -2.5],
                ["oppose", -2.0], ["protest", -2.0], ["grid strain", -2.0],
                ["brownout", -2.0], ["property value", -1.0], ["tax break", -0.5],
                ["jobs", 1.5], ["economic development", 1.5], ["investment", 1.0],
              ].map(([word, score]) => (
                <div key={word as string} className="flex justify-between">
                  <span className="text-[var(--muted)]">{word as string}</span>
                  <span className={`font-mono ${(score as number) < 0 ? "text-red-400" : "text-green-400"}`}>
                    {(score as number) > 0 ? "+" : ""}{score as number}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Vocal minority explanation */}
          <div className="mt-4 rounded-lg border border-amber-500/20 bg-amber-500/5 p-4">
            <h3 className="mb-2 text-sm font-semibold text-amber-400">Vocal Minority Correction</h3>
            <p className="text-xs text-[var(--muted)] leading-relaxed">
              500 angry posts in a county of 1,000,000 people is noise. 20 angry posts in a
              county of 2,000 people is a signal of organized opposition that can kill a project.
              We normalize by county population using a log-scaled posts-per-capita intensity
              factor. The final social resistance score = <code className="text-[var(--foreground)]">
              raw_sentiment × intensity_factor × volume_factor</code>.
            </p>
          </div>
        </Section>

        {/* ── Section 3: Feature Engineering ── */}
        <Section number={3} title="Feature Engineering">
          <p className="mb-4 text-sm text-[var(--muted)]">
            Before assigning weights, we check for multicollinearity (|r| &gt; 0.7) and
            normalize using z-score (for roughly normal features) or percentile rank
            (for skewed features with |skewness| &gt; 2). Min-max normalization is avoided
            because outliers like Loudoun County squash everything else to zero.
          </p>

          <div className="rounded-lg border border-[var(--card-border)] bg-[var(--card)] p-4">
            <h3 className="mb-2 text-xs font-semibold text-[var(--muted)]">NORMALIZATION STRATEGY</h3>
            <div className="space-y-2 text-xs">
              <div className="flex gap-3">
                <span className="rounded bg-blue-500/10 px-2 py-0.5 text-blue-400 font-mono">z-score</span>
                <span className="text-[var(--muted)]">|skewness| ≤ 2 → clip to [-3, +3], map to 0–100</span>
              </div>
              <div className="flex gap-3">
                <span className="rounded bg-purple-500/10 px-2 py-0.5 text-purple-400 font-mono">percentile</span>
                <span className="text-[var(--muted)]">|skewness| &gt; 2 → rank(pct=True) × 100</span>
              </div>
            </div>
            <p className="mt-3 text-xs text-[var(--muted)]">
              Correlation matrix heatmap saved to <code>outputs/correlation_matrix.png</code>.
              Highly correlated pairs (|r| &gt; 0.7) are flagged for removal or combination.
            </p>
          </div>
        </Section>

        {/* ── Section 4: Weight Derivation ── */}
        <Section number={4} title="Weight Derivation">
          <p className="mb-4 text-sm text-[var(--muted)]">
            Dimension weights are derived from logistic regression on historical site outcomes,
            not assumed arbitrarily. Class imbalance is handled with a four-layer defense.
          </p>

          {/* Class balance */}
          <div className="mb-4">
            <h3 className="mb-2 text-xs font-semibold text-[var(--muted)]">CLASS BALANCE</h3>
            <ClassBalance
              successful={validation.class_balance?.successful ?? null}
              blocked={validation.class_balance?.blocked ?? null}
              strategy={validation.imbalance_strategy}
            />
          </div>

          {/* F1 score */}
          <div className="mb-4 grid grid-cols-2 gap-3">
            <MetricCard
              label="Macro F1 Score"
              value={validation.macro_f1 != null ? validation.macro_f1.toFixed(3) : "—"}
              sub="Not accuracy — forces both-class performance"
            />
            <MetricCard
              label="Imbalance Layers"
              value="4"
              sub="balanced weights, SMOTE, stratified KFold, F1 eval"
            />
          </div>

          {/* Weight comparison */}
          <h3 className="mb-2 text-xs font-semibold text-[var(--muted)]">DERIVED vs ASSUMED WEIGHTS</h3>
          <WeightComparison derived={validation.weights} />
        </Section>

        {/* ── Section 5: Validation ── */}
        <Section number={5} title="Model Validation">
          <p className="mb-4 text-sm text-[var(--muted)]">
            Three statistical tests validate the PSI model against known site outcomes.
          </p>

          <div className="mb-4 grid grid-cols-2 gap-3 sm:grid-cols-3">
            <MetricCard
              label="AUC (ROC)"
              value={validation.auc != null ? validation.auc.toFixed(3) : "—"}
              sub="Area under ROC curve"
            />
            <MetricCard
              label="Mann-Whitney U"
              value={validation.social_test_p_value != null ? `p=${validation.social_test_p_value.toFixed(4)}` : "—"}
              sub={
                validation.social_test_p_value != null && validation.social_test_p_value < 0.05
                  ? "Significant at α=0.05"
                  : "Social resistance test"
              }
            />
            <MetricCard label="Validation Set" value={`${validation.n_validation_sites ?? "—"}`} sub="Known outcomes" />
          </div>

          {/* ROC curve image */}
          <div className="rounded-lg border border-[var(--card-border)] bg-[var(--card)] p-4 text-center">
            <h3 className="mb-2 text-xs font-semibold text-[var(--muted)]">ROC CURVE</h3>
            <p className="text-xs text-[var(--muted)]">
              Generated at <code>outputs/roc_curve.png</code> — displays true positive rate vs false positive
              rate for PSI-based viability prediction.
            </p>
          </div>

          {/* Sensitivity analysis */}
          {validation.sensitivity && validation.sensitivity.length > 0 && (
            <div className="mt-4">
              <h3 className="mb-2 text-xs font-semibold text-[var(--muted)]">
                SENSITIVITY ANALYSIS (Kendall&apos;s τ)
              </h3>
              <p className="mb-2 text-xs text-[var(--muted)]">
                Rankings compared across weight scenarios. τ &gt; 0.8 = stable, τ &lt; 0.5 = fragile.
              </p>
              <div className="overflow-hidden rounded-lg border border-[var(--card-border)]">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-[var(--card-border)] bg-[var(--card)]">
                      <th className="px-3 py-2 text-left text-[var(--muted)]">Scenario A</th>
                      <th className="px-3 py-2 text-left text-[var(--muted)]">Scenario B</th>
                      <th className="px-3 py-2 text-right text-[var(--muted)]">τ</th>
                      <th className="px-3 py-2 text-right text-[var(--muted)]">p-value</th>
                      <th className="px-3 py-2 text-right text-[var(--muted)]">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {validation.sensitivity.map((row, i) => (
                      <tr key={i} className="border-b border-[var(--card-border)] last:border-0">
                        <td className="px-3 py-2">{row.scenario_1}</td>
                        <td className="px-3 py-2">{row.scenario_2}</td>
                        <td className="px-3 py-2 text-right font-mono">{row.kendall_tau.toFixed(3)}</td>
                        <td className="px-3 py-2 text-right font-mono">{row.p_value.toFixed(4)}</td>
                        <td className="px-3 py-2 text-right">
                          <span className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${
                            row.stability === "STABLE"
                              ? "bg-green-500/10 text-green-400"
                              : row.stability === "FRAGILE"
                                ? "bg-red-500/10 text-red-400"
                                : "bg-yellow-500/10 text-yellow-400"
                          }`}>
                            {row.stability}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </Section>
      </div>
    </div>
  );
}

/* ── Reusable sub-components ── */

function Section({ number, title, children }: { number: number; title: string; children: React.ReactNode }) {
  return (
    <section>
      <div className="mb-4 flex items-center gap-3">
        <span className="flex h-7 w-7 items-center justify-center rounded-full bg-[var(--accent)]/10 text-sm font-bold text-[var(--accent)]">
          {number}
        </span>
        <h2 className="text-lg font-semibold">{title}</h2>
      </div>
      {children}
    </section>
  );
}

function MetricCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="rounded-lg border border-[var(--card-border)] bg-[var(--card)] p-3">
      <div className="text-lg font-bold font-mono">{value}</div>
      <div className="text-xs font-medium text-[var(--foreground)]">{label}</div>
      {sub && <div className="mt-0.5 text-[10px] text-[var(--muted)]">{sub}</div>}
    </div>
  );
}
