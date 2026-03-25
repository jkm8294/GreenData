"use client";

const FALLBACK = { power: 0.45, environmental: 0.3, social: 0.25 };

export default function WeightComparison({
  derived,
}: {
  derived: Record<string, number> | null;
}) {
  const dims = ["power", "environmental", "social"];

  return (
    <div className="overflow-hidden rounded-lg border border-[var(--card-border)]">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-[var(--card-border)] bg-[var(--card)]">
            <th className="px-4 py-2.5 text-left font-medium text-[var(--muted)]">Dimension</th>
            <th className="px-4 py-2.5 text-right font-medium text-[var(--muted)]">Assumed</th>
            <th className="px-4 py-2.5 text-right font-medium text-[var(--muted)]">Derived</th>
            <th className="px-4 py-2.5 text-right font-medium text-[var(--muted)]">Delta</th>
          </tr>
        </thead>
        <tbody>
          {dims.map((dim) => {
            const assumed = FALLBACK[dim as keyof typeof FALLBACK];
            const actual = derived?.[dim] ?? assumed;
            const delta = actual - assumed;
            return (
              <tr key={dim} className="border-b border-[var(--card-border)] last:border-0">
                <td className="px-4 py-2.5 font-medium capitalize">{dim}</td>
                <td className="px-4 py-2.5 text-right font-mono text-[var(--muted)]">
                  {assumed.toFixed(3)}
                </td>
                <td className="px-4 py-2.5 text-right font-mono font-semibold">
                  {actual.toFixed(3)}
                </td>
                <td className={`px-4 py-2.5 text-right font-mono ${delta > 0 ? "text-green-400" : delta < 0 ? "text-red-400" : "text-[var(--muted)]"}`}>
                  {delta > 0 ? "+" : ""}{delta.toFixed(3)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
