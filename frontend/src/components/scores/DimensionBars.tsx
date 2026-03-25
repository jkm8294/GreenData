"use client";

interface Props {
  power: number;
  environmental: number;
  social: number;
}

function Bar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-sm">
        <span className="text-[var(--muted)]">{label}</span>
        <span className="font-mono font-medium">{value.toFixed(1)}</span>
      </div>
      <div className="h-2.5 rounded-full bg-[var(--card-border)]">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${Math.min(value, 100)}%`, backgroundColor: color }}
        />
      </div>
    </div>
  );
}

export default function DimensionBars({ power, environmental, social }: Props) {
  return (
    <div className="space-y-3">
      <Bar label="Power Infrastructure" value={power} color="#3b82f6" />
      <Bar label="Environmental" value={environmental} color="#22c55e" />
      <Bar label="Social Resistance" value={social} color="#f59e0b" />
    </div>
  );
}
