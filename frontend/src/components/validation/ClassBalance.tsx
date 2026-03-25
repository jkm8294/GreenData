"use client";

interface Props {
  successful: number | null;
  blocked: number | null;
  strategy: string | null;
}

export default function ClassBalance({ successful, blocked, strategy }: Props) {
  const total = (successful || 0) + (blocked || 0);
  const successPct = total > 0 ? ((successful || 0) / total) * 100 : 0;
  const blockedPct = total > 0 ? ((blocked || 0) / total) * 100 : 0;

  return (
    <div className="space-y-3">
      <div className="flex gap-2 text-sm">
        <div className="flex-1 rounded-lg border border-[var(--card-border)] bg-[var(--card)] p-3 text-center">
          <div className="text-2xl font-bold text-green-400">{successful ?? "—"}</div>
          <div className="text-xs text-[var(--muted)]">Successful</div>
        </div>
        <div className="flex-1 rounded-lg border border-[var(--card-border)] bg-[var(--card)] p-3 text-center">
          <div className="text-2xl font-bold text-red-400">{blocked ?? "—"}</div>
          <div className="text-xs text-[var(--muted)]">Blocked</div>
        </div>
        <div className="flex-1 rounded-lg border border-[var(--card-border)] bg-[var(--card)] p-3 text-center">
          <div className="text-2xl font-bold">{total || "—"}</div>
          <div className="text-xs text-[var(--muted)]">Total</div>
        </div>
      </div>

      {/* Proportion bar */}
      {total > 0 && (
        <div className="flex h-3 overflow-hidden rounded-full">
          <div
            className="bg-green-500 transition-all"
            style={{ width: `${successPct}%` }}
            title={`${successPct.toFixed(0)}% successful`}
          />
          <div
            className="bg-red-500 transition-all"
            style={{ width: `${blockedPct}%` }}
            title={`${blockedPct.toFixed(0)}% blocked`}
          />
        </div>
      )}

      {strategy && (
        <div className="rounded-md bg-blue-500/10 px-3 py-2 text-xs text-blue-400">
          <span className="font-semibold">Imbalance Strategy: </span>
          {strategy}
        </div>
      )}

      <p className="text-xs text-[var(--muted)]">
        With {total > 0 ? `a ${((successful || 0) / Math.max(blocked || 1, 1)).toFixed(1)}:1 ratio` : "unknown ratio"},
        accuracy alone is misleading. A model that always predicts &quot;Successful&quot; achieves{" "}
        {successPct.toFixed(0)}% accuracy for free. We evaluate on <strong className="text-[var(--foreground)]">Macro F1</strong> to
        force the model to perform well on <em>both</em> classes.
      </p>
    </div>
  );
}
